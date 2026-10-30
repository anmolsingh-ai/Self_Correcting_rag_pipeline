import os
import traceback
import uuid

# --- ZeroGPU startup check -------------------------------------------------
# The Space hardware is ZeroGPU, which refuses to start unless at least one
# function is decorated with @spaces.GPU. This pipeline does NOT need a GPU
# (Gemini runs on Google's servers, the MiniLM models are tiny), so we register
# a placeholder that is never called. `respond` itself runs on normal CPU.
# `spaces` must be imported BEFORE main.py loads torch / the models.
try:
    import spaces
except ImportError:  # running locally, no `spaces` package
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _gpu_placeholder():
        return "ok"

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


def respond(message, history, session_id):
    try:
        resp = chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )
        return str(getattr(resp, "content", resp))
    except Exception as e:
        traceback.print_exc()  # full traceback goes to the container logs
        return f"Error: {type(e).__name__}: {e}"


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