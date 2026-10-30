import os
import uuid

# --- ZeroGPU support -------------------------------------------------------
# `spaces` must be imported BEFORE main.py loads torch / the models.
# On Hugging Face ZeroGPU it provides the @spaces.GPU decorator; when running
# locally (no `spaces` package) we fall back to a no-op decorator.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # local: reads GEMINI_API_KEY from your .env file

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it to .env (local) or to your "
        "host's secrets/environment variables (e.g. Hugging Face Space secrets)."
    )

from main import build_pipeline

# Built once at startup (replaces @st.cache_resource)
chain = build_pipeline(os.path.join("data", "Notes.pdf"))


@gpu
def respond(message, history, session_id):
    resp = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    return str(getattr(resp, "content", resp))


with gr.Blocks(title="Self-Correcting RAG Chat") as demo:
    gr.Markdown("# Self-Correcting RAG Chat")

    # Per-user session id (replaces st.session_state["sid"])
    session_id = gr.State(lambda: str(uuid.uuid4()))

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[session_id],
        textbox=gr.Textbox(placeholder="Ask about the document"),
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)