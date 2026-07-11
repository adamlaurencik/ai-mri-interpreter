import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

MODEL = "gpt-5.6-sol"
TEMPERATURE = None
MAX_TOKENS = 10000

# gpt-5.6-sol pricing (USD per token), standard tier.
# Full table per 1M tokens: input $5.00, cached input $0.50, cache writes $6.25,
# output $30.00. estimate_cost() only models plain input/output below.
PRICE_INPUT_PER_TOKEN = 5.00 / 1_000_000
PRICE_OUTPUT_PER_TOKEN = 30.00 / 1_000_000

HIGHLIGHT_COLOR = "#ffd166"

UI_LABELS = {
    "English": {"translation": "Translation", "context": "Context"},
    "Slovak":  {"translation": "Preklad",     "context": "Kontext"},
    "Czech":   {"translation": "Překlad",     "context": "Kontext"},
    "German":  {"translation": "Übersetzung", "context": "Kontext"},
}


def _ui_label(key: str, language: str) -> str:
    return UI_LABELS.get(language, UI_LABELS["English"]).get(key, key)


def _normalize_certainty(raw: str) -> str:
    """Collapse to {certain, uncertain}. Defaults to 'certain' when missing/unknown input."""
    v = (raw or "").strip().lower()
    if v in ("uncertain", "probable", "possible", "unknown"):
        return "uncertain"
    return "certain"

st.markdown(
    """
This is a spike tool. It does not diagnose or replace medical professionals.
Only use de-identified MRI text.
"""
)

# ---------------------------------------------------------------------------
# Test examples — drop .txt files into examples/. Filename (without .txt) is
# the label shown in the dropdown.
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES_SENTINEL = "— select an example —"


def _load_examples() -> dict[str, str]:
    examples: dict[str, str] = {EXAMPLES_SENTINEL: ""}
    if EXAMPLES_DIR.is_dir():
        for path in sorted(EXAMPLES_DIR.glob("*.txt")):
            examples[path.stem] = path.read_text(encoding="utf-8")
    return examples


_examples = _load_examples()


def _apply_example() -> None:
    choice = st.session_state.get("example_choice", EXAMPLES_SENTINEL)
    st.session_state["mri_text"] = _examples.get(choice, "")


st.selectbox(
    "Load a test example",
    list(_examples.keys()),
    key="example_choice",
    on_change=_apply_example,
    help=f"Reads .txt files from {EXAMPLES_DIR.name}/. You can still edit the text after loading.",
)
if len(_examples) == 1:
    st.caption(f"_No examples found. Drop .txt files into `{EXAMPLES_DIR.name}/` to populate this list._")

mri_text = st.text_area(
    "MRI report text",
    height=320,
    key="mri_text",
    placeholder="Paste the MRI report text here...",
)

run = st.button("Interpret report", type="primary")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ReportLine:
    """One non-empty line of the original report, with its character offset."""
    line_id: int
    start: int
    text: str


@dataclass
class FindingSegment:
    exact_quote: str
    line_id: int = -1  # from Step 1; the line the quote belongs to
    start: int = -1
    end: int = -1
    match_score: float = 0.0


@dataclass
class FindingGroup:
    """One clinical finding at one location (the leaf level). May have several
    mentions/segments across the report."""
    group_id: str
    finding_type: str           # canonical English from Step 1 (kept for debugging/grouping invariants)
    anatomical_location: str    # canonical English from Step 1
    certainty: str              # enum: certain / uncertain (internal only; not shown in UI)
    segments: list[FindingSegment]
    category_id: str = ""        # coarse cluster slug (finding kind, location-independent), from Step 1
    category: str = ""           # human-readable category label (English, from Step 1)


@dataclass
class Category:
    """A cluster of same-kind findings across levels — the unit of explanation
    and display. One card per category in the UI; one shared badge number in the
    highlighted report."""
    category_id: str
    label: str                          # English label from Step 1
    groups: list[FindingGroup]
    number: int = 0                     # badge number shared by all this category's spans
    label_localized: str = ""           # Step 2: category label in target language
    levels_summary: str = ""            # Step 2: short localized phrase listing levels + severity
    translation: str = ""               # Step 2: one plain-language sentence for the whole category
    context: str = ""                   # Step 2: 1–2 sentences on what the category is


# ---------------------------------------------------------------------------
# Report normalization — un-wrap soft (margin) line breaks
# ---------------------------------------------------------------------------
# PDF/print exports wrap long sentences at a fixed margin, so a single finding
# can straddle a hard newline (e.g. "...prominent prevertebral\nspondylosis").
# That breaks per-line quote anchoring: the LLM cannot quote the phrase from one
# physical line, so it either fragments the finding or the fuzzy match degrades.
# We undo those margin wraps before anything else. Each dropped newline is
# replaced 1:1 with a space so character offsets are preserved; real section
# headers and sentence breaks are kept.

_WRAP_MIN_LEN = 80  # a physical line this long likely ran to the page margin

_STRUCTURAL_START = re.compile(
    r"""^\s*(
        [A-Z][0-9]+-[A-Z]?[0-9]+\s*:        # level marker, e.g. "C4-5:", "C7-T1:"
        | \d+\.\s                            # numbered impression item, e.g. "1. "
        | [A-Z][A-Z\ \-/]{2,}:?\s*$          # ALL-CAPS section header, e.g. "DISCS:"
    )""",
    re.VERBOSE,
)


def _continues_previous(prev: str, cur: str) -> bool:
    """True when `cur` is a soft-wrapped continuation of `prev`, not a new line."""
    if not prev.strip() or not cur.strip():
        return False
    if _STRUCTURAL_START.match(cur):
        return False
    if len(prev.rstrip()) < _WRAP_MIN_LEN:
        return False
    return prev.rstrip()[-1] not in ".:;!?"


def unwrap_soft_wraps(report_text: str) -> str:
    """Join margin-wrapped physical lines into logical lines (offset-preserving)."""
    text = report_text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for line in text.split("\n"):
        if out and _continues_previous(out[-1], line):
            out[-1] = out[-1] + " " + line  # dropped "\n" -> " " keeps length/offsets
        else:
            out.append(line)
    return "\n".join(out)


def split_report_into_lines(report_text: str) -> list[ReportLine]:
    """Split on newlines, skip blank lines. line_id is 1-based and sequential."""
    lines: list[ReportLine] = []
    offset = 0
    line_id = 1
    for raw in report_text.split("\n"):
        if raw.strip():
            lines.append(ReportLine(line_id=line_id, start=offset, text=raw))
            line_id += 1
        offset += len(raw) + 1  # +1 for the \n
    return lines


def format_numbered_report(lines: list[ReportLine]) -> str:
    return "\n".join(f"[{l.line_id}] {l.text}" for l in lines)


# ---------------------------------------------------------------------------
# Step 1 — LLM extraction
# ---------------------------------------------------------------------------

SEGMENTATION_PROMPT = """\
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

When the SAME clinical finding at the SAME anatomical location is mentioned more \
than once in the report, extract EACH mention as a separate object AND assign them \
the SAME "group_id". Distinct findings (different type or different location) MUST \
have different group_ids. When group_ids match, the finding_type, anatomical_location, \
and certainty fields MUST also match across those objects.

Additionally, assign each finding a coarser "category_id": a slug for the KIND of \
finding, IGNORING anatomical location and severity. Findings of the same kind at \
different levels or of different severity MUST share the same "category_id" (e.g. \
mild stenosis at C4-5 and severe stenosis at C6-7 are both "spinal-canal-stenosis"). \
Different kinds of finding MUST have different category_ids. Do NOT use a fixed list \
— name the categories from what you actually see in THIS report, but keep the naming \
consistent across all findings of the same kind. A "category_id" is always coarser \
than a "group_id": every group_id maps to exactly one category_id. Also provide a \
short human-readable "category" label that matches the category_id.

The report below is presented with each line prefixed by a bracketed line number, \
e.g. "[3] ...". For EVERY finding you extract, return the line number of the line \
that contains the quote as the integer "line_id". The "exact_quote" MUST be a \
verbatim sub-phrase drawn from that same line.

For EACH finding return a JSON object with exactly these keys:
- "line_id": integer line number (taken from the bracketed prefix) of the line \
that contains this finding's exact_quote
- "exact_quote": the verbatim phrase or sub-phrase from that line that describes \
this finding (copy it character-for-character from within that line, do not paraphrase, \
do not include the "[N] " prefix)
- "group_id": a short stable slug combining finding type and location, e.g. \
"disc-herniation-l4-l5", "marrow-signal-change-l3-body". Mentions of the same \
finding at the same location MUST share this exact slug. Use only lowercase \
letters, digits, and hyphens.
- "category_id": a coarser slug for the KIND of finding, ignoring location and \
severity, e.g. "spinal-canal-stenosis", "neural-foraminal-narrowing". All findings \
of the same kind share this slug. Use only lowercase letters, digits, and hyphens.
- "category": a short human-readable label matching category_id, e.g. \
"spinal canal stenosis", "neural foraminal narrowing".
- "finding_type": short category label, e.g. "disc herniation", "stenosis", \
"signal change", "structural anomaly", "degenerative change", etc.
- "anatomical_location": the anatomical region or level described, spelled out \
in full (e.g. "L4/5 intervertebral disc" not "L4/5")
- "certainty": one of "certain" or "uncertain". Default to "certain"; use \
"uncertain" only when the report language itself hedges (e.g. "možné", \
"pravdepodobne", "suspektný", "cannot exclude").

Return ONLY a JSON object with a single key "findings" whose value is an array \
of the above objects. No markdown fences, no extra text.

MRI report:
{numbered_report}
"""


EXPLANATION_PROMPT = """\
You are a radiology assistant. Below is an MRI report followed by findings that \
were already extracted from it, grouped into CATEGORIES. Each category is one KIND \
of finding (e.g. spinal canal stenosis) and lists the levels where it occurs, each \
with its severity. For EACH category as a whole, produce patient-facing text in \
{language} for a non-expert audience. Use the full report as context, but do NOT \
introduce findings that are not in the list.

Write ONE explanation per category that covers ALL of its levels together — do NOT \
repeat near-identical text per level. For every category, produce all of the \
following IN {language}:
1. "category" — a short translation of the category label into {language}.
2. "levels_summary" — a short phrase listing the affected levels and their severity, \
e.g. "C4-5, C5-6 a C6-7 – mierne" (levels stay as-is, e.g. "C4-5"; severity words in {language}).
3. "translation" — ONE sentence that re-states, in plain non-medical language, what \
the report says about this category across its levels.
4. "context" — 1–2 sentences describing what this kind of finding IS in plain terms. \
Describe the current state of things, not predictions. Be concrete but not alarmist.

IMPORTANT WORDING RULES for the "translation" and "context" fields:
- Do NOT use language that frames the body as deteriorating from use, such as \
"wear and tear", "drying out", "wearing out", "worn down", "degenerated from \
use", or equivalents in {language}. Such phrasing can make patients think that \
moving makes things worse and that they should move less. \
Instead describe these as "natural" or "degenerative" changes.
- Do NOT predict prognosis. Avoid describing how the finding "may worsen", "could \
get worse over time", or what problems it "could lead to / cause in the future". \
Describe what is — what the finding is, or what it may be / will be — not what \
might happen to it later.

ALL output fields MUST be written in {language}. Do not leave any field \
in English or in the original report language unless {language} IS English.

Return ONLY a JSON object with a single key "explanations" whose value is an \
array of objects, each with exactly these keys:
- "category_id": the exact category_id from the input list, copied verbatim
- "category": translation of the input category label into {language}
- "levels_summary": short {language} phrase listing the levels and severity
- "translation": 1 sentence in {language} — plain-language rendering of the category
- "context": 1–2 sentences in {language} — plain description of what the finding is (no prognosis, no "wear and tear" language)

No markdown fences, no extra text.

MRI report:
{report}

Categories:
{findings_json}
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
            timeout=120,
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


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    return cleaned


def _coerce_line_id(raw_value) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def extract_findings(lines: list[ReportLine]) -> tuple[list[FindingGroup], str, str]:
    """Step 1: LLM segmentation — extract structured findings (no explanations yet)."""
    prompt = SEGMENTATION_PROMPT.format(numbered_report=format_numbered_report(lines))
    raw, usage = call_openai_raw(prompt)
    cost_str = estimate_cost(usage)

    try:
        parsed = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Segmentation call returned invalid JSON: {exc}",
            response_body=raw,
        ) from exc

    groups: dict[str, FindingGroup] = {}
    ungrouped_counter = 0
    for f in parsed.get("findings", []):
        gid = f.get("group_id") or ""
        if not gid:
            gid = f"__ungrouped_{ungrouped_counter}"
            ungrouped_counter += 1
        segment = FindingSegment(
            exact_quote=f.get("exact_quote", ""),
            line_id=_coerce_line_id(f.get("line_id")),
        )
        if gid in groups:
            groups[gid].segments.append(segment)
        else:
            groups[gid] = FindingGroup(
                group_id=gid,
                finding_type=f.get("finding_type", ""),
                anatomical_location=f.get("anatomical_location", ""),
                certainty=_normalize_certainty(f.get("certainty", "")),
                segments=[segment],
                category_id=f.get("category_id", "") or "",
                category=f.get("category", "") or "",
            )
    return list(groups.values()), raw, cost_str


def generate_explanations(
    report_text: str, categories: list[Category], language: str
) -> tuple[list[Category], str, str]:
    """Step 2: one LLM call that fills in patient-facing text per CATEGORY.

    Explaining per category (rather than per level) dedupes the near-identical
    text that repeats when the same finding appears at many spinal levels.
    """
    if not categories:
        return categories, "", estimate_cost({})

    payload = [
        {
            "category_id": c.category_id,
            "category": c.label,
            "levels": [
                {
                    "anatomical_location": g.anatomical_location,
                    "finding_type": g.finding_type,
                    "certainty": g.certainty,
                    "quote": g.segments[0].exact_quote if g.segments else "",
                }
                for g in c.groups
            ],
        }
        for c in categories
    ]
    prompt = EXPLANATION_PROMPT.format(
        language=language,
        report=report_text,
        findings_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    raw, usage = call_openai_raw(prompt)
    cost_str = estimate_cost(usage)

    try:
        parsed = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Explanation call returned invalid JSON: {exc}",
            response_body=raw,
        ) from exc

    by_id = {c.category_id: c for c in categories}
    for item in parsed.get("explanations", []):
        c = by_id.get(item.get("category_id"))
        if c is None:
            continue
        c.label_localized = item.get("category", "")
        c.levels_summary = item.get("levels_summary", "")
        c.translation = item.get("translation", "")
        c.context = item.get("context", "")
    return categories, raw, cost_str


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


def map_findings_to_offsets(
    report_text: str, lines: list[ReportLine], groups: list[FindingGroup]
) -> list[FindingGroup]:
    """Locate each segment's exact_quote within its line, falling back to the full report."""
    line_by_id = {l.line_id: l for l in lines}
    for g in groups:
        for seg in g.segments:
            quote = seg.exact_quote.strip()
            if not quote:
                continue
            line = line_by_id.get(seg.line_id)
            if line is not None:
                rel_start, rel_end, score = _fuzzy_locate(line.text, quote)
                if rel_start >= 0:
                    seg.start = line.start + rel_start
                    seg.end = line.start + rel_end
                    seg.match_score = score
                    continue
            # Fallback: line_id missing/invalid, or quote not found within that line
            start, end, score = _fuzzy_locate(report_text, quote)
            seg.start = start
            seg.end = end
            seg.match_score = score
    return groups


# ---------------------------------------------------------------------------
# Step 3 — HTML highlighted text
# ---------------------------------------------------------------------------

CERTAINTY_RANK = {"certain": 1, "uncertain": 0}


def build_categories(groups: list[FindingGroup]) -> list[Category]:
    """Cluster groups by category_id into Category objects, preserving
    first-appearance order and assigning each a 1-based badge number.

    Groups without a category_id each form their own singleton category, so an
    uncategorized finding still renders as a standalone card, as before.
    """
    order: list[str] = []
    buckets: dict[str, list[FindingGroup]] = {}
    for g in groups:
        cid = g.category_id or f"__uncat_{id(g)}"
        if cid not in buckets:
            buckets[cid] = []
            order.append(cid)
        buckets[cid].append(g)

    categories: list[Category] = []
    for number, cid in enumerate(order, 1):
        gs = buckets[cid]
        label = gs[0].category or gs[0].finding_type
        categories.append(Category(category_id=cid, label=label, groups=gs, number=number))
    return categories


def _build_label_array(report_text: str, groups: list[FindingGroup]) -> list[int | None]:
    """Return a per-character index into groups (highest-certainty group wins)."""
    label: list[int | None] = [None] * len(report_text)
    for idx, g in enumerate(groups):
        rank = CERTAINTY_RANK.get(g.certainty, 0)
        for seg in g.segments:
            if seg.start < 0:
                continue
            for pos in range(seg.start, min(seg.end, len(report_text))):
                existing = label[pos]
                if existing is None or CERTAINTY_RANK.get(groups[existing].certainty, 0) < rank:
                    label[pos] = idx
    return label


def _finding_span(text: str, number: int, tooltip_label: str) -> str:
    color = HIGHLIGHT_COLOR
    tooltip = html.escape(f"#{number} {tooltip_label}")
    badge = (
        f'<sup style="background:#444;color:#fff;border-radius:3px;'
        f'padding:0 3px;font-size:0.65rem;margin-right:1px">{number}</sup>'
    )
    return (
        f'<span style="background:{color};border-radius:3px;padding:1px 4px;'
        f'outline:2px solid {color};outline-offset:0px;" '
        f'title="{tooltip}">{badge}{html.escape(text)}</span>'
    )


def build_highlighted_html(report_text: str, categories: list[Category]) -> str:
    """
    Step 3: build an HTML string with <span> highlights for every matched segment.
    All segments belonging to the same category share the category's badge number
    and a single highlight color. Overlapping spans are merged by priority
    (highest certainty wins).
    """
    # Flatten to the group list the label array indexes into, and remember each
    # group's category number + label for the badge/tooltip.
    groups = [g for c in categories for g in c.groups]
    meta_by_group_idx = {
        id(g): (c.number, c.label_localized or c.label)
        for c in categories
        for g in c.groups
    }
    label = _build_label_array(report_text, groups)

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
            g = groups[idx]
            number, tip = meta_by_group_idx[id(g)]
            parts.append(_finding_span(report_text[i:j], number, tip))
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

    # Un-wrap PDF-style margin line breaks so findings that straddle a wrap
    # become quotable from a single logical line. Offset-preserving, so this is
    # the canonical text for matching and display from here on.
    report_text = unwrap_soft_wraps(mri_text)
    report_lines = split_report_into_lines(report_text)

    # ---- Step 1 — Segmentation ----
    with st.status("Step 1 — Segmentation...", expanded=True) as step1_status:
        try:
            groups, raw_segment, cost_segment = extract_findings(report_lines)
            total_segments = sum(len(g.segments) for g in groups)
            step1_status.update(
                label=(
                    f"Step 1 — Segmentation complete "
                    f"({len(groups)} findings, {total_segments} mentions) · {cost_segment}"
                ),
                state="complete",
            )
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
        st.code(raw_segment, language="json")

    # Cluster the per-location findings into categories (the display + explanation
    # unit). Category order = first appearance, and fixes the badge numbering used
    # by both the highlighted report and the Findings column.
    categories = build_categories(groups)

    # ---- Step 2 — Explanation ----
    with st.status("Step 2 — Explanation...", expanded=True) as step2_status:
        try:
            categories, raw_explain, cost_explain = generate_explanations(
                report_text, categories, target_language
            )
            step2_status.update(
                label=(
                    f"Step 2 — Explanation complete "
                    f"({sum(1 for c in categories if c.translation)}/{len(categories)} categories explained) · {cost_explain}"
                ),
                state="complete",
            )
        except LLMError as exc:
            step2_status.update(label=f"Step 2 — failed: {exc}", state="error")
            if exc.status_code:
                st.error(f"HTTP {exc.status_code}")
            if exc.response_body:
                st.code(exc.response_body, language="json")
            st.stop()
        except Exception as exc:
            step2_status.update(label="Step 2 — unexpected error", state="error")
            st.exception(exc)
            st.stop()

    with st.expander("Raw LLM output (Step 2)", expanded=False):
        st.code(raw_explain or "(skipped — no findings to explain)", language="json")

    # ---- Step 3 — Fuzzy back-mapping ----
    with st.status("Step 3 — Fuzzy back-mapping...", expanded=True) as step3_status:
        groups = map_findings_to_offsets(report_text, report_lines, groups)
        all_segments = [seg for g in groups for seg in g.segments]
        matched = sum(1 for seg in all_segments if seg.start >= 0)
        step3_status.update(
            label=f"Step 3 — Back-mapping complete ({matched}/{len(all_segments)} quotes located)",
            state="complete",
        )

    with st.expander("Mapping results (Step 3)", expanded=False):
        mapping_rows = [
            {
                "category_id": g.category_id,
                "group_id": g.group_id,
                "line_id": seg.line_id,
                "quote": seg.exact_quote[:60] + ("…" if len(seg.exact_quote) > 60 else ""),
                "start": seg.start,
                "end": seg.end,
                "match_score": round(seg.match_score, 1),
                "located": seg.start >= 0,
            }
            for g in groups
            for seg in g.segments
        ]
        st.dataframe(mapping_rows, use_container_width=True)

    # ---- Step 4 — Interpretation ----
    st.subheader("Step 4 — Interpretation")

    col_text, col_findings = st.columns([1, 1], gap="large")

    with col_text:
        st.markdown("**Highlighted report**")

        highlighted = build_highlighted_html(report_text, categories)
        st.markdown(highlighted, unsafe_allow_html=True)

    with col_findings:
        st.markdown("**Findings**")
        translation_label = _ui_label("translation", target_language)
        context_label = _ui_label("context", target_language)

        # One card per category. Multi-level categories are summarized in a single
        # card (levels_summary line); a single-level category reads as a plain card.
        for c in categories:
            label = c.label_localized or c.label
            if c.levels_summary:
                levels_note = f' &nbsp;<span style="color:#888;font-size:0.85rem">{html.escape(c.levels_summary)}</span>'
            elif len(c.groups) > 1:
                levels_note = f' &nbsp;<span style="color:#888;font-size:0.85rem">{len(c.groups)} levels</span>'
            else:
                levels_note = ""
            with st.container(border=True):
                st.markdown(
                    f'<span style="background:#444;color:#fff;border-radius:3px;padding:1px 6px;'
                    f'font-size:0.8rem;margin-right:6px">#{c.number}</span>'
                    f'<strong>{html.escape(label)}</strong>{levels_note}',
                    unsafe_allow_html=True,
                )
                if c.translation:
                    st.markdown(f"**{translation_label}:** {c.translation}")
                if c.context:
                    st.markdown(f"**{context_label}:** {c.context}")
                # Collapse source phrases by wording. The same phrasing recurs
                # across levels and impression items (e.g. "flattening of the
                # ventral thecal sac" at C4-5/C5-6/C6-7 and again in the summary);
                # those are distinct physical mentions but identical text, so we
                # show each unique wording once with an ×N count. This also folds
                # away the same-span duplicates from summarizing phrases.
                phrases: dict[str, dict] = {}
                order: list[str] = []
                for g in c.groups:
                    for seg in g.segments:
                        norm = seg.exact_quote.strip().lower()
                        if not norm:
                            continue
                        if norm not in phrases:
                            phrases[norm] = {"text": seg.exact_quote, "count": 0, "located": False}
                            order.append(norm)
                        phrases[norm]["count"] += 1
                        if seg.start >= 0:
                            phrases[norm]["located"] = True
                with st.expander(f"Source phrases ({len(order)})", expanded=False):
                    for norm in order:
                        info = phrases[norm]
                        clipped = info["text"][:80] + ("…" if len(info["text"]) > 80 else "")
                        suffix = f" ×{info['count']}" if info["count"] > 1 else ""
                        if info["located"]:
                            st.caption(f'"{clipped}"{suffix}')
                        else:
                            st.caption(f'_(not located)_ "{clipped}"{suffix}')

    st.caption(f"Completed at {datetime.now().strftime('%H:%M:%S')}")
