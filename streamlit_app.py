import os
import uuid
import streamlit as st
import boto3
from botocore.exceptions import ClientError

AGENT_ID = os.getenv("AGENT_ID", "NCTQZKII1Y").strip()         # e.g. A1B2C3...
ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "GDKCKJIYXI").strip()   # e.g. TSTALIASID
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

STREAM_FINAL = os.getenv("STREAM_FINAL_RESPONSE", "true").lower() == "true"
GUARDRAIL_INTERVAL = int(os.getenv("APPLY_GUARDRAIL_INTERVAL", "50"))

@st.cache_resource(show_spinner=False)
def bedrock_agent_client():
    # Uses env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (/ AWS_SESSION_TOKEN)
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

def invoke_agent_stream(prompt: str, session_id: str, session_attrs: dict | None = None, baseline_prompt_attrs: dict | None = None):
    """
    Calls Agents for Bedrock and yields chunks as they arrive.
    Sends session_attrs into sessionState.sessionAttributes AND promptSessionAttributes when provided.
    If overrides are OFF, we still pass a tiny 'baseline_prompt_attrs' (brand/channel/lang) so Lambda can map to headers.
    """
    client = bedrock_agent_client()

    # -- line before --
    kwargs = dict(
        agentId=AGENT_ID,
        agentAliasId=ALIAS_ID,
        sessionId=session_id,
        inputText=prompt,
        enableTrace=False,  # set True if you want traces
        streamingConfigurations={
            "applyGuardrailInterval": GUARDRAIL_INTERVAL,
            "streamFinalResponse": STREAM_FINAL,
        },
    )
    # Send both sticky+turn attrs when overrides are ON
    if session_attrs:
        kwargs["sessionState"] = {
            "sessionAttributes": session_attrs,           # sticky
            "promptSessionAttributes": session_attrs,     # turn-scoped
        }
    # When overrides are OFF, still provide minimal promptSessionAttributes so Lambda can inject headers
    elif baseline_prompt_attrs:
        kwargs["sessionState"] = {
            "promptSessionAttributes": baseline_prompt_attrs
        }
    resp = client.invoke_agent(**kwargs)
    # -- line after --

    completion = resp.get("completion")
    if completion is None:
        # Fallback: return whole response if the SDK ever changes behavior
        yield resp.get("outputText", "")
        return

    for event in completion:
        if "chunk" in event:
            yield event["chunk"]["bytes"].decode("utf-8", errors="ignore")

# -- KaTeX-safe Markdown helper --
def escape_katex(md: str) -> str:
    return (
        md.replace("$", r"\$")
          .replace(r"\(", r"\\(")
          .replace(r"\)", r"\\)")
          .replace(r"\[", r"\\[")
          .replace(r"\]", r"\\]")
    )

st.set_page_config(page_title="Bedrock Agent Chat", page_icon="🤖", layout="centered")

# ---------- Prefill & state ----------
DEFAULT_CUST = os.getenv("DEFAULT_CUSTOMER_OUID", "1E5A1F564E180BD3EBF02D7D5007DB28")
DEFAULT_BRAND = os.getenv("DEFAULT_X_BRAND", "DEMO-DEMO")
DEFAULT_CHANNEL = os.getenv("DEFAULT_X_CHANNEL", "AGENT_TOOL")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en")

if "overrides" not in st.session_state:
    st.session_state.overrides = {
        "jwt": "",
        "customerOuid": DEFAULT_CUST,
        "billingAccountOuid": "",
        "parentOuid": "",
        "offeringOuid": "",
        "specOuid": "",
        "msisdn": "",
        "goodwillSizeGb": 2,
        "goodwillReason": "boosterOrPassRefund",
        "lang": DEFAULT_LANG,
        "xBrand": DEFAULT_BRAND,
        "xChannel": DEFAULT_CHANNEL,
    }
if "use_overrides" not in st.session_state:
    st.session_state.use_overrides = False

# =======================
# ⚙️ UNIFIED SETTINGS UI
# =======================
# -- line before --
with st.expander("⚙️ Settings", expanded=False):
    st.caption("Configure everything here. Toggle overrides to send your params as session attributes; otherwise minimal defaults are sent as prompt attributes.")
    # Mode toggle
    st.session_state.use_overrides = st.checkbox(
        "Use these overrides in calls (send as sessionAttributes & promptSessionAttributes)",
        value=st.session_state.use_overrides,
        help="ON → send your fields. OFF → send tiny defaults (brand/channel/lang) so Lambda can inject headers.",
    )

    st.divider()
    st.subheader("🔐 Auth")
    st.caption("Leave JWT empty to rely on Lambda’s STATIC_JWT")
    st.session_state.overrides["jwt"] = st.text_input(
        "JWT (include 'Bearer …' if you want to override)",
        value=st.session_state.overrides.get("jwt", ""),
        placeholder="Bearer eyJhbGciOiJIUzUxMiJ9.…",
    )

    st.divider()
    st.subheader("👤 Customer context")
    st.caption("This customerOuid is used by ALL calls. For DTO without an ID, Lambda will use this first; if empty, it falls back to the Lambda default.")
    st.session_state.overrides["customerOuid"] = st.text_input(
        "customerOuid",
        value=st.session_state.overrides.get("customerOuid", ""),
        placeholder=DEFAULT_CUST,
    )

    st.divider()
    st.subheader("🌐 Call context")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.overrides["xBrand"] = st.text_input(
            "X-Brand",
            value=st.session_state.overrides.get("xBrand", DEFAULT_BRAND),
            placeholder=DEFAULT_BRAND,
        )
    with c2:
        st.session_state.overrides["xChannel"] = st.text_input(
            "X-Channel",
            value=st.session_state.overrides.get("xChannel", DEFAULT_CHANNEL),
            placeholder=DEFAULT_CHANNEL,
        )
    with c3:
        st.session_state.overrides["lang"] = st.selectbox(
            "Accept-Language",
            options=["en", "fr"],
            index=0 if (st.session_state.overrides.get("lang", DEFAULT_LANG) == "en") else 1,
        )

    st.divider()
    st.subheader("🎁 Goodwill parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.overrides["parentOuid"] = st.text_input(
            "parentOuid (subscription as PARENT)",
            value=st.session_state.overrides.get("parentOuid", ""),
            placeholder="Subscription OUID if you want to pin the line",
        )
        st.session_state.overrides["specOuid"] = st.text_input(
            "specOuid",
            value=st.session_state.overrides.get("specOuid", ""),
            placeholder="8B3C73498520F7048BC00F449DBAE447",
        )
        st.session_state.overrides["msisdn"] = st.text_input(
            "msisdn (prefer this to let Lambda auto-resolve the line)",
            value=st.session_state.overrides.get("msisdn", ""),
            placeholder="0613423341",
        )
    with col2:
        st.session_state.overrides["billingAccountOuid"] = st.text_input(
            "billingAccountOuid",
            value=st.session_state.overrides.get("billingAccountOuid", ""),
            placeholder="B5C34D432E67E1543F02255AACB06057",
        )
        st.session_state.overrides["offeringOuid"] = st.text_input(
            "offeringOuid",
            value=st.session_state.overrides.get("offeringOuid", ""),
            placeholder="47F2CE64A772B870D62F5BD19ED02196",
        )
        st.session_state.overrides["goodwillSizeGb"] = st.number_input(
            "goodwillSizeGb (GB)",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.overrides.get("goodwillSizeGb", 2)),
            step=1,
        )
        st.session_state.overrides["goodwillReason"] = st.text_input(
            "goodwillReason",
            value=st.session_state.overrides.get("goodwillReason", "boosterOrPassRefund"),
            placeholder="refund / boosterOrPassRefund …",
        )
# -- line after --

def build_session_attributes() -> dict:
    """
    Build sessionAttributes sent to Bedrock.
    - If 'Use these overrides' is OFF -> return {} (lambda defaults apply; we still send minimal prompt attrs elsewhere).
    - If ON -> include only non-empty fields.
    Critical: customerOuid is global (applies to both DTO and goodwill).
    """
    if not st.session_state.use_overrides:
        return {}
    o = st.session_state.overrides
    attrs = {}

    # Global
    if o.get("jwt"):            attrs["jwt"] = o["jwt"]
    if o.get("customerOuid"):   attrs["customerOuid"] = o["customerOuid"]
    if o.get("lang"):           attrs["lang"] = o["lang"]
    if o.get("xBrand"):         attrs["xBrand"] = o["xBrand"]
    if o.get("xChannel"):       attrs["xChannel"] = o["xChannel"]

    # Goodwill-specific (optional)
    if o.get("billingAccountOuid"):  attrs["billingAccountOuid"] = o["billingAccountOuid"]
    if o.get("parentOuid"):          attrs["parentOuid"] = o["parentOuid"]
    if o.get("offeringOuid"):        attrs["offeringOuid"] = o["offeringOuid"]
    if o.get("specOuid"):            attrs["specOuid"] = o["specOuid"]
    if o.get("msisdn"):              attrs["msisdn"] = o["msisdn"]

    # Always include goodwill size/reason if overrides enabled (safe defaults)
    try:
        size = int(o.get("goodwillSizeGb", 2))
        if size < 1: size = 1
        attrs["goodwillSizeGb"] = str(size)
    except Exception:
        attrs["goodwillSizeGb"] = "2"
    if o.get("goodwillReason"):
        attrs["goodwillReason"] = o["goodwillReason"]

    return attrs

st.title("🤖 AI Assistant")
st.caption("Type below to talk to your AI Assistant.")

if not AGENT_ID or not ALIAS_ID:
    st.error("Missing AGENT_ID or AGENT_ALIAS_ID environment variables.")
    st.stop()

# Keep a stable chat session per browser session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(escape_katex(m["content"]))

# --- Quick prompts (click to auto-ask) ---
quick_prompts = [
    "🧾 Summarize account",
    "💳 Check billing & payments",
    "📊 Analyze consumption; recommend plan/booster",
    "🚨 Spot risks; suggest actions",
    "🎟️ Review tickets; propose next steps",
]

cols = st.columns(len(quick_prompts))
for i, label in enumerate(quick_prompts):
    if cols[i].button(label, use_container_width=True, key=f"qp_{i}"):
        q = st.session_state.get("_queued_user_msgs", [])
        q.append(label)
        st.session_state["_queued_user_msgs"] = q
        st.rerun()

# Always render the chat input so it never disappears
typed_input = st.chat_input("Ask me anything…")

# Use queued quick prompt first, otherwise typed input
q = st.session_state.get("_queued_user_msgs", [])
if q:
    user_input = q.pop(0)
    st.session_state["_queued_user_msgs"] = q
else:
    user_input = typed_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(escape_katex(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""
        try:
            # -- line before --
            session_attrs = build_session_attributes()

            # Build baseline prompt attributes when overrides are OFF
            baseline_prompt_attrs = None
            if not st.session_state.use_overrides:
                baseline_prompt_attrs = {
                    "xBrand": DEFAULT_BRAND,
                    "xChannel": DEFAULT_CHANNEL,
                    "lang": DEFAULT_LANG,
                }

            with st.expander("🧪 Debug — what will be sent", expanded=False):
                st.write({
                    "sessionId": st.session_state.session_id,
                    "agentId": AGENT_ID,
                    "aliasId": ALIAS_ID,
                    "use_overrides": st.session_state.use_overrides,
                    "sessionAttributes": session_attrs,              # sticky+turn (when ON)
                    "promptDefaultsWhenOff": baseline_prompt_attrs,  # turn-only (when OFF)
                })

            for chunk in invoke_agent_stream(
                user_input,
                st.session_state.session_id,
                session_attrs=session_attrs if session_attrs else None,
                baseline_prompt_attrs=baseline_prompt_attrs
            ):
                acc += chunk
                placeholder.markdown(escape_katex(acc) + "▌")
            # -- line after --
        except ClientError as e:
            acc = f"⚠️ AWS error: {e}"
        except Exception as e:
            acc = f"⚠️ Runtime error: {e}"
        placeholder.markdown(escape_katex(acc))
    st.session_state.messages.append({"role": "assistant", "content": acc})
