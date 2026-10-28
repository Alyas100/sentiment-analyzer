"""ZeroGPU Gradio dashboard for Malay/Manglish sentiment analysis."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "aisingapore/Gemma-SEA-LION-v3-9B"
ADAPTER = "alyaspronoob/malay-sentiment-gemma-sealion-9b-lora"
LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
MAX_ROWS = 50

ASPECT_KEYWORDS = {
    "Delivery": ("delivery", "post", "courier", "shipped", "arrive", "sampai", "penghantaran"),
    "Packaging": ("packaging", "pack", "bungkus", "kotak", "kemas", "kemik"),
    "Price": ("price", "harga", "murah", "mahal", "value", "worth"),
    "Quality": ("quality", "kualiti", "barang", "product", "item", "rosak"),
    "Customer service": ("customer", "service", "seller", "chat", "reply", "refund"),
    "Size / fit": ("size", "saiz", "fit", "muat", "besar", "kecil"),
}

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = None


def load_model():
    """Load the model only after ZeroGPU allocates CUDA."""

    global model
    if model is None:
        base = AutoModelForCausalLM.from_pretrained(
            BASE,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, ADAPTER, subfolder="lora_gemma9b")
        model = model.merge_and_unload()
        model.to("cuda")
        model.eval()
    return model


def make_prompt(text: str) -> str:
    instruction = (
        "Analyze the sentiment of the following Malay/Manglish text. "
        "Classify it as POSITIVE, NEGATIVE, or NEUTRAL."
    )
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text[:600]}\n\n### Response:\n"


def _classify_text(text: str) -> str:
    active_model = load_model()
    encoded = tokenizer(make_prompt(text), return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = active_model.generate(
            **encoded,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = (
        tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        .strip()
        .upper()
    )
    return next(
        (label for label in LABELS if generated.startswith(label)), f"UNKNOWN ({generated})"
    )


@spaces.GPU(duration=60)
def classify(text: str) -> str:
    """Public single-text endpoint, kept for external clients."""

    if not text or not text.strip():
        return "Please enter some text."
    return _classify_text(text)


def _read_rows(file_path: str) -> list[dict[str, str]]:
    with Path(file_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The CSV is empty.")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"Please upload at most {MAX_ROWS} reviews per analysis.")
    normalized = []
    for index, row in enumerate(rows, 1):
        lowered = {
            str(key).strip().lower().replace(" ", "_"): value or "" for key, value in row.items()
        }
        text = (
            lowered.get("review_text") or lowered.get("text") or lowered.get("review") or ""
        ).strip()
        if text:
            normalized.append(
                {
                    "review_id": lowered.get("review_id") or lowered.get("id") or str(index),
                    "date": lowered.get("date") or lowered.get("review_date") or "",
                    "platform": lowered.get("platform") or "Unknown",
                    "review_text": text,
                }
            )
    if not normalized:
        raise ValueError("CSV must contain a review_text, text, or review column.")
    return normalized


def _aspects(text: str) -> str:
    lower = text.lower()
    return (
        ", ".join(
            name
            for name, words in ASPECT_KEYWORDS.items()
            if any(re.search(r"\b" + re.escape(word) + r"\b", lower) for word in words)
        )
        or "—"
    )


def _keywords(rows: list[dict[str, str]]) -> list[str]:
    stopwords = {"the", "and", "yang", "dan", "itu", "ini", "sangat", "tak", "gila", "je", "pun"}
    words = Counter()
    for row in rows:
        if row["sentiment"] == "Negative":
            words.update(
                word
                for word in re.findall(r"[a-zA-Z0-9'-]{3,}", row["review_text"].lower())
                if word not in stopwords
            )
    return [word for word, _ in words.most_common(8)]


def _results_table(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [
            row["review_id"],
            row["date"],
            row["platform"],
            row["review_text"],
            row["sentiment"],
            row["aspects"],
        ]
        for row in rows
    ]


@spaces.GPU(duration=60)
def analyze_csv(file_path: str | None, platform: str, with_aspects: bool):
    if not file_path:
        raise gr.Error("Please upload a CSV file first.")
    try:
        rows = _read_rows(file_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    if platform != "All Shops":
        rows = [row for row in rows if row["platform"].strip().lower() == platform.lower()]
    if not rows:
        raise gr.Error("No reviews match the selected platform.")

    for row in rows:
        row["sentiment"] = _classify_text(row["review_text"]).capitalize()
        row["aspects"] = _aspects(row["review_text"]) if with_aspects else "—"

    counts = {
        label.capitalize(): sum(row["sentiment"] == label.capitalize() for row in rows)
        for label in LABELS
    }
    total = len(rows)
    negative_ratio = counts["Negative"] / total if total else 0
    metrics = f"""
    <div class="metric-grid">
      <div class="metric-card total"><span>Reviews analyzed</span><strong>{total}</strong><small>Current selection</small></div>
      <div class="metric-card positive"><span>Positive</span><strong>{counts["Positive"]}</strong><small>Customer sentiment</small></div>
      <div class="metric-card negative"><span>Negative</span><strong>{counts["Negative"]}</strong><small>Needs attention</small></div>
      <div class="metric-card neutral"><span>Neutral</span><strong>{counts["Neutral"]}</strong><small>{negative_ratio:.1%} negative ratio</small></div>
    </div>
    """
    bars = "".join(
        f'<div class="bar-row"><span>{label}</span><div class="bar {label.lower()}" style="width:{counts[label] / total * 100:.1f}%"></div><b>{counts[label]}</b></div>'
        for label in ("Positive", "Negative", "Neutral")
    )
    keywords = (
        " ".join(f'<span class="keyword">{word}</span>' for word in _keywords(rows))
        or "<em>No negative-review keywords.</em>"
    )
    insights = f"""
    <div class="insight-grid">
      <div class="chart-card"><h3>Sentiment distribution</h3>{bars}</div>
      <div class="chart-card"><h3>Top complaint keywords</h3><div class="keywords">{keywords}</div></div>
    </div>
    """
    summary = f"### Analysis complete\nUsing the **Gemma + LoRA** model on **{platform}**. The Space analyzed **{total}** reviews."
    return summary, metrics + insights, _results_table(rows)


CSS = """
:root { --ink:#172033; --muted:#667085; --line:#e7eaf0; --brand:#f97316; --bg:#f7f8fb; }
.gradio-container { max-width:1180px !important; margin:auto; background:var(--bg); }
.brand { display:flex; align-items:center; gap:18px; padding:18px 0 8px; }
.mark { background:var(--brand); color:#fff; border-radius:14px; font-weight:800; font-size:24px; padding:16px 12px; box-shadow:0 8px 20px #f9731638; }
.eyebrow { color:#c2410c; font-size:11px; font-weight:800; letter-spacing:1.6px; }
.brand h1 { color:var(--ink); font-size:34px; margin:2px 0; letter-spacing:-1px; }
.brand p, .intro { color:var(--muted); }
.panel { border:1px solid var(--line); border-radius:16px; background:#fff; padding:10px 16px 16px; box-shadow:0 6px 22px #1720330a; }
.panel h2 { color:var(--ink); }
.primary { background:var(--brand) !important; border-color:var(--brand) !important; }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:14px 0; }
.metric-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:17px 18px; border-top:4px solid var(--brand); }
.metric-card.positive { border-top-color:#16a34a; } .metric-card.negative { border-top-color:#dc2626; } .metric-card.neutral { border-top-color:#64748b; }
.metric-card span,.metric-card small { display:block; color:var(--muted); font-size:13px; } .metric-card strong { display:block; color:var(--ink); font-size:30px; margin:6px 0; }
.insight-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; } .chart-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px; min-height:170px; }
.chart-card h3 { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
.bar-row { display:grid; grid-template-columns:75px 1fr 30px; gap:10px; align-items:center; margin:13px 0; color:var(--muted); font-size:13px; } .bar { height:12px; border-radius:8px; min-width:2px; }
.bar.positive { background:#16a34a; } .bar.negative { background:#dc2626; } .bar.neutral { background:#64748b; }
.keywords { display:flex; flex-wrap:wrap; gap:8px; } .keyword { background:#fff1eb; color:#c2410c; padding:7px 10px; border-radius:999px; font-size:13px; }
@media (max-width:760px) { .metric-grid,.insight-grid { grid-template-columns:repeat(2,1fr); } .brand h1 { font-size:27px; } }
"""


with gr.Blocks(title="Review sentiment, made actionable.", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        '<div class="brand"><div class="mark">SA</div><div><div class="eyebrow">INSIGHT STUDIO</div>'
        "<h1>Review sentiment, made actionable.</h1><p>Understand what customers love, where they struggle, and why.</p></div></div>"
    )
    gr.Markdown(
        "Analyze code-mixed Malay/English e-commerce reviews with sentiment, aspect insights, "
        "and exportable reports.",
        elem_classes="intro",
    )
    with gr.Group(elem_classes="panel"):
        gr.Markdown("### Start an analysis")
        with gr.Row():
            file_input = gr.File(
                label="Review dataset", file_types=[".csv"], type="filepath", scale=2
            )
            with gr.Column(scale=1):
                platform = gr.Dropdown(
                    ["All Shops", "Shopee", "TikTok", "Lazada"], value="All Shops", label="Platform"
                )
                aspects = gr.Checkbox(
                    True, label="Aspect-based analysis", info="Delivery, price, quality..."
                )
        analyze = gr.Button("Analyze reviews", variant="primary", elem_classes="primary")
    gr.Markdown("### Snapshot")
    summary = gr.Markdown()
    metrics = gr.HTML()
    table = gr.Dataframe(
        headers=["Review ID", "Date", "Platform", "Review", "Sentiment", "Aspects"],
        datatype=["str"] * 6,
        label="Results",
        wrap=True,
    )
    analyze.click(
        analyze_csv, [file_input, platform, aspects], [summary, metrics, table], api_name="analyze"
    )
    with gr.Accordion("Single review check", open=False):
        single_text = gr.Textbox(label="Malay / Manglish text", lines=3)
        single_output = gr.Textbox(label="Sentiment", interactive=False)
        single_button = gr.Button("Classify")
        single_button.click(classify, single_text, single_output, api_name="predict")

if __name__ == "__main__":
    demo.launch()
