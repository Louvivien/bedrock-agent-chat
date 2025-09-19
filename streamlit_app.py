import os
import uuid
import streamlit as st
import boto3
from botocore.exceptions import ClientError

AGENT_ID = os.getenv("AGENT_ID", "").strip()         # e.g. A1B2C3...
ALIAS_ID = os.getenv("AGENT_ALIAS_ID", "").strip()   # e.g. TSTALIASID
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

STREAM_FINAL = os.getenv("STREAM_FINAL_RESPONSE", "true").lower() == "true"
GUARDRAIL_INTERVAL = int(os.getenv("APPLY_GUARDRAIL_INTERVAL", "50"))

@st.cache_resource(show_spinner=False)
def bedrock_agent_client():
    # Uses env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (/ AWS_SESSION_TOKEN)
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

def invoke_agent_stream(prompt: str, session_id: str, session_attrs: dict | None = None):
    """
    Calls Agents for Bedrock and yields chunks as they arrive.
    Uses 'invoke_agent' event stream per AWS docs.
    If session_attrs is provided, it is sent under sessionState.sessionAttributes.
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
    # Attach session attributes only if we actually have overrides
    if session_attrs:
        kwargs["sessionState"] = {"sessionAttributes": session_attrs}
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

# ---------- Collapsed parameter panels (prefilled & collapsed) ----------
# We keep values in session_state so they persist across interactions.
if "overrides" not in st.session_state:
    st.session_state.overrides = {
        "jwt": "",                    # leave empty to use Lambda STATIC_JWT
        # goodwill params (leave empty to let Lambda defaults win)
        "customerOuid": "",
        "billingAccountOuid": "",
        "parentOuid": "",
        "offeringOuid": "",
        "specOuid": "",
        "msisdn": "",
        "goodwillSizeGb": 2,
        "goodwillReason": "boosterOrPassRefund",
    }
if "use_overrides" not in st.session_state:
    st.session_state.use_overrides = False

with st.expander("🔐 Auth (JWT token) — collapsed by default", expanded=False):
    st.caption("Leave empty to rely on Lambda’s STATIC_JWT")
    st.session_state.overrides["jwt"] = st.text_input(
        "JWT (include 'Bearer ...' if you want to override)",
        value=st.session_state.overrides.get("jwt", ""),
        placeholder="Bearer eyJhbGciOiJIUzUxMiJ9.…",
    )

with st.expander("🎁 Goodwill parameters — collapsed by default", expanded=False):
    st.caption("Fill any you want to override. Empty fields are ignored so Lambda defaults still apply.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.overrides["customerOuid"] = st.text_input(
            "customerOuid",
            value=st.session_state.overrides.get("customerOuid", ""),
            placeholder="8B0B07E340842EDF73EBD6DE1208C1C3",
        )
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
            "msisdn (prefer this if you want Lambda to auto-resolve the line)",
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

st.session_state.use_overrides = st.checkbox(
    "Use these overrides in calls (send as sessionAttributes)",
    value=st.session_state.use_overrides,
    help="When enabled, only non-empty fields are sent; empty fields are omitted so Lambda defaults win."
)

# Helper: build the sessionAttributes dict with only non-empty values
def build_session_attributes() -> dict:
    if not st.session_state.use_overrides:
        return {}
    o = st.session_state.overrides
    attrs = {}
    # Map exactly to what the Lambda expects in sessionAttributes
    if o.get("jwt"):                   attrs["jwt"] = o["jwt"]
    if o.get("customerOuid"):          attrs["customerOuid"] = o["customerOuid"]
    if o.get("billingAccountOuid"):    attrs["billingAccountOuid"] = o["billingAccountOuid"]
    if o.get("parentOuid"):            attrs["parentOuid"] = o["parentOuid"]
    if o.get("offeringOuid"):          attrs["offeringOuid"] = o["offeringOuid"]
    if o.get("specOuid"):              attrs["specOuid"] = o["specOuid"]
    if o.get("msisdn"):                attrs["msisdn"] = o["msisdn"]
    # Always send goodwillSizeGb / goodwillReason if overrides enabled
    try:
        size = int(o.get("goodwillSizeGb", 2))
        if size < 1: size = 1
        attrs["goodwillSizeGb"] = str(size)
    except Exception:
        attrs["goodwillSizeGb"] = "2"
    if o.get("goodwillReason"):
        attrs["goodwillReason"] = o["goodwillReason"]
    return attrs

st.title("🤖 Bedrock Agent Chat (Streamlit)")
st.caption("Type below to talk to your Bedrock Agent. Deployed on Render.")

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
            for chunk in invoke_agent_stream(user_input, st.session_state.session_id, session_attrs):
                acc += chunk
                placeholder.markdown(escape_katex(acc) + "▌")
            # -- line after --
        except ClientError as e:
            acc = f"⚠️ AWS error: {e}"
        except Exception as e:
            acc = f"⚠️ Runtime error: {e}"
        placeholder.markdown(escape_katex(acc))
    st.session_state.messages.append({"role": "assistant", "content": acc})
