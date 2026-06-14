import gradio as gr

from .config import Config
from .embeddings import load_model
from .pipeline import run

_CONFIG = Config()
_MODEL = None  # lazy-loaded on first request


def _get_model(path: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(path)
    return _MODEL


def format_result(res, embedding_dim: int):
    """Return (headline_poem, markdown_table)."""
    headline = " ".join(res.output_words) if res.output_words else "(no words embedded)"
    lines = [
        f"**Rank achieved:** {res.rank_achieved} of {embedding_dim}"
        + ("" if res.reached_dim else "  _(document too short to reach full dimension)_"),
        "",
        "| # | reduced → word | distance |",
        "|---|---|---|",
    ]
    for i, (w, d) in enumerate(zip(res.output_words, res.distances)):
        lines.append(f"| {i} | {w} | {d:.4f} |")
    lines += ["", f"**Input words embedded:** {', '.join(res.input_words)}"]
    return headline, "\n".join(lines)


def _run(text, uploaded, embedding_dim, delta, scale):
    content = text or ""
    if uploaded:
        with open(uploaded, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
    cfg = Config(embedding_dim=int(embedding_dim), delta=float(delta), scale=int(scale))
    res = run(content, _get_model(cfg.model_path), cfg)
    return format_result(res, cfg.embedding_dim)


def build_ui():
    with gr.Blocks(title="BasisOfUnderstanding") as demo:
        gr.Markdown("# BasisOfUnderstanding\nLLL lattice reduction over word embeddings.")
        with gr.Row():
            text = gr.Textbox(label="Paste text", lines=10)
            uploaded = gr.File(label="...or upload .txt", file_types=[".txt"], type="filepath")
        with gr.Accordion("Advanced", open=False):
            embedding_dim = gr.Dropdown([100, 300], value=100, label="EMBEDDING_DIM")
            delta = gr.Slider(0.26, 0.999, value=0.99, step=0.001, label="δ (Lovász)")
            scale = gr.Number(value=10**6, label="SCALE", precision=0)
        run_btn = gr.Button("Run", variant="primary")
        headline = gr.Textbox(label="Decoded poem", interactive=False)
        table = gr.Markdown()
        run_btn.click(
            _run,
            inputs=[text, uploaded, embedding_dim, delta, scale],
            outputs=[headline, table],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
