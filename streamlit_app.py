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
    # Uses env AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (/ AWS_SESSION_TOKEN) set in Render
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

def invoke_agent_stream(prompt: str, session_id: str):
    """
    Calls Agents for Bedrock and yields chunks as they arrive.
    Uses 'invoke_agent' event stream per AWS docs.
    """
    client = bedrock_agent_client()
    resp = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=ALIAS_ID,
        sessionId=session_id,
        inputText=prompt,
        enableTrace=False,  # set True if you want traces
        streamingConfigurations={
            "applyGuardrailInterval": GUARDRAIL_INTERVAL,
            "streamFinalResponse": STREAM_FINAL
        }
    )
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
    "📊 Analyze consumption; recommend plan/booster",  # <- updated label
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
            for chunk in invoke_agent_stream(user_input, st.session_state.session_id):
                acc += chunk
                placeholder.markdown(escape_katex(acc) + "▌")
        except ClientError as e:
            acc = f"⚠️ AWS error: {e}"
        except Exception as e:
            acc = f"⚠️ Runtime error: {e}"
        placeholder.markdown(escape_katex(acc))
    st.session_state.messages.append({"role": "assistant", "content": acc})
