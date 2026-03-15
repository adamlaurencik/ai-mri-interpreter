import html
import json
import os
from dataclasses import dataclass
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

st.set_page_config(
    page_title="MRI Interpretation Spike",
    page_icon="🧠",
    layout="wide",
)


def require_password() -> None:
    app_password = os.getenv("APP_PASSWORD")
    if not app_password:
        st.error("APP_PASSWORD is not set in the environment.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("MRI Interpretation Spike")
    st.caption("Enter the password to continue")
    with st.form("password_gate"):
        st.text_input("Password", type="password", key="password_input")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if st.session_state.get("password_input") == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


if (
    os.getenv("APP_PASSWORD")
    and os.getenv("USE_PASSWORD_AUTH", "false").lower() == "true"
):
    require_password()

st.title("MRI Interpretation Spike")
st.caption("Paste MRI report text and get a structured interpretation summary")

with st.sidebar:
    st.header("Settings")
    target_language = st.selectbox(
        "Output language",
        ["English", "Slovak", "Czech", "German"],
        index=0,
    )
    st.caption("Set OPENAI_API_KEY in your environment")

MODEL = "gpt-5.4"
TEMPERATURE = 0.2
MAX_TOKENS = 10000

# gpt-5.4 pricing (USD per token)
PRICE_INPUT_PER_TOKEN = 2.50 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 15.00 / 1_000_000

# Color palette per certainty level
CERTAINTY_COLORS = {
    "definite": "#ffd166",
    "probable": "#a8d8ea",
    "possible": "#c3f0ca",
    "unknown": "#e0e0e0",
}

st.markdown(
    """
This is a spike tool. It does not diagnose or replace medical professionals.
Only use de-identified MRI text.
"""
)

mri_text = st.text_area(
    "MRI report text",
    height=320,
    placeholder="Paste the MRI report text here...",
)

run = st.button("Interpret report", type="primary")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    exact_quote: str
    finding_type: str
    anatomical_location: str
    certainty: str
    explanation: str
    start: int = -1
    end: int = -1
    match_score: float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — LLM extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a radiology assistant specializing in spinal imaging. Analyze the MRI 
report below and extract every finding that represents a departure from 
ideal/healthy anatomy.

CRITICAL INSTRUCTIONS: Do not filter out "mild," "degenerative," or "age-related" 
findings. Even if a finding is described as "dystrophic," "chronic," or 
"incidental," it must be extracted if it describes a change in bone marrow 
signal, vertebral structure, disc health, or nerve space. If it could 
theoretically contribute to a patient's pain or discomfort, include it.
Filter findings that describe normal anatomy. 
Keep in mind it can happen that the sentence is negated and thus is not a positive finding.
Even if some findings are repeated, include them multiple times.
Make sure to split findings into individual findings, even if they are next to each other in the text.

For EACH finding return a JSON object with exactly these keys:
- "exact_quote": the verbatim phrase or sentence from the report that describes \
this finding (copy it character-for-character, do not paraphrase)
- "finding_type": short category label, e.g. "disc herniation", "stenosis", \
"signal change", "structural anomaly", "degenerative change", etc.
- "anatomical_location": the anatomical region or level described, spelled out \
in full (e.g. "L4/5 intervertebral disc" not "L4/5")
- "certainty": one of "definite", "probable", "possible" — based on the \
language used in the report
- "explanation": 1–2 sentences explaining this finding in plain language for \
a non-expert patientx

Return ONLY a JSON object with a single key "findings" whose value is an array \
of the above objects. No markdown fences, no extra text.

Output language for "explanation": {language}

MRI report:
{report}
"""


class LLMError(Exception):
    """Raised when the OpenAI API call fails. Carries structured context."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def call_openai_raw(prompt: str) -> tuple[str, dict]:
    """Returns (output_text, usage_dict). Raises LLMError on any failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set in the environment.")

    payload = {
        "model": MODEL,
        "input": prompt,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_TOKENS,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=60,
        )
    except requests.exceptions.Timeout:
        raise LLMError("Request timed out after 60 s. The model may be overloaded.")
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(f"Network error — could not reach OpenAI API: {exc}")

    if not response.ok:
        raise LLMError(
            f"API returned HTTP {response.status_code}",
            status_code=response.status_code,
            response_body=response.text,
        )

    data = response.json()

    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    output_text += block.get("text", "")

    if not output_text:
        raise LLMError(
            "API responded OK but returned no output text.",
            status_code=response.status_code,
            response_body=response.text,
        )
    return output_text, data.get("usage", {})


def estimate_cost(usage: dict) -> str:
    input_tok = usage.get("input_tokens", 0)
    output_tok = usage.get("output_tokens", 0)
    cost = input_tok * PRICE_INPUT_PER_TOKEN + output_tok * PRICE_OUTPUT_PER_TOKEN
    return f"{input_tok} in / {output_tok} out tokens — ~${cost:.4f}"


def extract_findings(report_text: str, language: str) -> tuple[list[Finding], str, str]:
    """Step 1: call LLM and parse the structured JSON response. Returns (findings, raw, cost_str)."""
    prompt = EXTRACTION_PROMPT.format(language=language, report=report_text)
    raw, usage = call_openai_raw(prompt)
    cost_str = estimate_cost(usage)

    # Strip accidental markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model returned invalid JSON: {exc}",
            response_body=cleaned,
        ) from exc
    findings = [
        Finding(
            exact_quote=f.get("exact_quote", ""),
            finding_type=f.get("finding_type", ""),
            anatomical_location=f.get("anatomical_location", ""),
            certainty=f.get("certainty", "unknown").lower(),
            explanation=f.get("explanation", ""),
        )
        for f in parsed.get("findings", [])
    ]
    return findings, raw, cost_str


# ---------------------------------------------------------------------------
# Step 2 — Fuzzy back-mapping
# ---------------------------------------------------------------------------

_BOUNDARY_SLACK = 4  # max chars the LLM may have added/removed at each boundary


def _fuzzy_locate(report_text: str, quote: str) -> tuple[int, int, float]:
    """
    Return (start, end, score) for the best match of quote inside report_text.

    Strategy:
      1. Exact (case-sensitive) — perfect when the LLM copied verbatim.
      2. Exact (case-insensitive) — catches casing differences.
      3. Step-1 sliding window at fixed length — finds the best start position
         without any coarse-step gaps that could skip the true position.
      4. Boundary refinement ±SLACK — tries slightly shorter/longer windows
         around the best start to recover a char the LLM added or dropped at
         either end (the "missing first/last letter" class of bug).
      Note: step-1 scan is fast enough for typical MRI report lengths.
    """
    quote_len = len(quote)
    n = len(report_text)
    quote_lower = quote.lower()
    text_lower = report_text.lower()

    # 1. Exact (case-sensitive)
    pos = report_text.find(quote)
    if pos != -1:
        return pos, pos + quote_len, 100.0

    # 2. Exact (case-insensitive)
    pos = text_lower.find(quote_lower)
    if pos != -1:
        return pos, pos + quote_len, 99.0

    # 3. Step-1 scan — guaranteed to find the globally best fixed-length window
    best_score = 0.0
    best_start = 0
    for i in range(0, max(1, n - quote_len + 1)):
        score = fuzz.ratio(quote_lower, text_lower[i : i + quote_len])
        if score > best_score:
            best_score = score
            best_start = i

    # 4. Boundary refinement — vary both start and end by ±SLACK around best_start
    best_end = best_start + quote_len
    slack = _BOUNDARY_SLACK
    lo = max(0, best_start - slack)
    hi = min(n, best_start + slack + 1)
    for start in range(lo, hi):
        for end in range(
            max(start + 1, start + quote_len - slack),
            min(n + 1, start + quote_len + slack + 1),
        ):
            score = fuzz.ratio(quote_lower, text_lower[start:end])
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    if best_score >= 60:
        return best_start, best_end, best_score
    return -1, -1, best_score


def map_findings_to_offsets(report_text: str, findings: list[Finding]) -> list[Finding]:
    """Step 2: locate each exact_quote in report_text and store character offsets."""
    for f in findings:
        quote = f.exact_quote.strip()
        if not quote:
            continue
        start, end, score = _fuzzy_locate(report_text, quote)
        f.start = start
        f.end = end
        f.match_score = score
    return findings


# ---------------------------------------------------------------------------
# Step 3 — HTML highlighted text
# ---------------------------------------------------------------------------

CERTAINTY_RANK = {"definite": 3, "probable": 2, "possible": 1, "unknown": 0}


def _build_label_array(report_text: str, findings: list[Finding]) -> list[int | None]:
    """Return a per-character index into findings (highest-certainty finding wins)."""
    label: list[int | None] = [None] * len(report_text)
    for idx, f in enumerate(findings):
        if f.start < 0:
            continue
        rank = CERTAINTY_RANK.get(f.certainty, 0)
        for pos in range(f.start, min(f.end, len(report_text))):
            existing = label[pos]
            if existing is None or CERTAINTY_RANK.get(findings[existing].certainty, 0) < rank:
                label[pos] = idx
    return label


def _finding_span(text: str, finding: Finding, number: int) -> str:
    color = CERTAINTY_COLORS.get(finding.certainty, "#e0e0e0")
    tooltip = html.escape(f"#{number} {finding.finding_type} | {finding.anatomical_location} | {finding.certainty}")
    badge = (
        f'<sup style="background:#444;color:#fff;border-radius:3px;'
        f'padding:0 3px;font-size:0.65rem;margin-right:1px">{number}</sup>'
    )
    return (
        f'<span style="background:{color};border-radius:3px;padding:1px 4px;'
        f'outline:2px solid {color};outline-offset:0px;" '
        f'title="{tooltip}">{badge}{html.escape(text)}</span>'
    )


def build_highlighted_html(report_text: str, findings: list[Finding]) -> str:
    """
    Step 3: build an HTML string with <span> highlights for every matched finding.
    Overlapping spans are merged by priority (highest certainty wins).
    """
    label = _build_label_array(report_text, findings)

    parts: list[str] = []
    i = 0
    while i < len(report_text):
        idx = label[i]
        j = i + 1
        if idx is None:
            while j < len(report_text) and label[j] is None:
                j += 1
            parts.append(html.escape(report_text[i:j]))
        else:
            while j < len(report_text) and label[j] == idx:
                j += 1
            parts.append(_finding_span(report_text[i:j], findings[idx], idx + 1))
        i = j

    body = "".join(parts).replace("\n", "<br>")
    return f'<div style="font-family:monospace;line-height:1.8;font-size:0.95rem">{body}</div>'


# ---------------------------------------------------------------------------
# Main UI — run pipeline
# ---------------------------------------------------------------------------

if run:
    if not mri_text.strip():
        st.warning("Paste MRI report text before running.")
        st.stop()

    st.divider()

    # ---- Step 1 ----
    with st.status("Step 1 — LLM extraction...", expanded=True) as step1_status:
        try:
            findings, raw_llm, cost_str = extract_findings(mri_text, target_language)
            step1_status.update(label=f"Step 1 — LLM extraction complete ({len(findings)} findings) · {cost_str}", state="complete")
        except LLMError as exc:
            step1_status.update(label=f"Step 1 — failed: {exc}", state="error")
            if exc.status_code:
                st.error(f"HTTP {exc.status_code}")
            if exc.response_body:
                st.code(exc.response_body, language="json")
            st.stop()
        except Exception as exc:
            step1_status.update(label="Step 1 — unexpected error", state="error")
            st.exception(exc)
            st.stop()

    with st.expander("Raw LLM output (Step 1)", expanded=False):
        st.code(raw_llm, language="json")

    # ---- Step 2 ----
    with st.status("Step 2 — Fuzzy back-mapping...", expanded=True) as step2_status:
        findings = map_findings_to_offsets(mri_text, findings)
        matched = sum(1 for f in findings if f.start >= 0)
        step2_status.update(
            label=f"Step 2 — Back-mapping complete ({matched}/{len(findings)} quotes located)",
            state="complete",
        )

    with st.expander("Mapping results (Step 2)", expanded=False):
        mapping_rows = [
            {
                "quote": f.exact_quote[:60] + ("…" if len(f.exact_quote) > 60 else ""),
                "start": f.start,
                "end": f.end,
                "match_score": round(f.match_score, 1),
                "located": f.start >= 0,
            }
            for f in findings
        ]
        st.dataframe(mapping_rows, use_container_width=True)

    # ---- Step 3 ----
    st.subheader("Step 3 — Interpretation")

    col_text, col_findings = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("**Highlighted report**")

        # Legend
        legend_html = " ".join(
            f'<span style="background:{color};border-radius:3px;padding:2px 8px;margin-right:6px">'
            f'{certainty}</span>'
            for certainty, color in CERTAINTY_COLORS.items()
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown("")

        highlighted = build_highlighted_html(mri_text, findings)
        st.markdown(highlighted, unsafe_allow_html=True)

    with col_findings:
        st.markdown("**Findings**")
        for i, f in enumerate(findings, 1):
            color = CERTAINTY_COLORS.get(f.certainty, "#e0e0e0")
            with st.container(border=True):
                st.markdown(
                    f'<span style="background:#444;color:#fff;border-radius:3px;padding:1px 6px;'
                    f'font-size:0.8rem;margin-right:6px">#{i}</span>'
                    f'<span style="background:{color};border-radius:3px;padding:1px 6px;'
                    f'font-size:0.8rem">{f.certainty}</span> &nbsp;'
                    f'<strong>{f.finding_type}</strong> — {f.anatomical_location}',
                    unsafe_allow_html=True,
                )
                st.markdown(f.explanation)
                if f.start >= 0:
                    st.caption(f'"{f.exact_quote[:80]}{"…" if len(f.exact_quote) > 80 else ""}"')
                else:
                    st.caption("_(quote not located in source text)_")

    st.caption(f"Completed at {datetime.now().strftime('%H:%M:%S')}")
