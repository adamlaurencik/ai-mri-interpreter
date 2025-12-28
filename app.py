import json
import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

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


if os.getenv("APP_PASSWORD") and os.getenv("USE_PASSWORD_AUTH", "false").lower() == "true":
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
    model = st.text_input("Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens", 256, 2000, 800, 64)
    st.caption("Set OPENAI_API_KEY in your environment")

st.markdown(
    """
This is a spike tool. It does not diagnose or replace medical professionals.
Only use de-identified MRI text.
"""
)

left, right = st.columns([2, 1])

with left:
    mri_text = st.text_area(
        "MRI report text",
        height=320,
        placeholder="Paste the MRI report text here...",
    )

    run = st.button("Interpret report", type="primary")

with right:
    st.subheader("What you will get")
    st.write(
        "- main findings\n- likely issues\n- follow-up suggestions (if mentioned)\n- short plain-language summary"
    )

status = st.empty()


def build_prompt(report_text: str, language: str) -> str:
    return (
        "You are a radiology assistant translating an MRI report for a layperson. "
        "Explain the findings in simple, non-technical language, without medical jargon. "
        "Be clear about what the report says and what it does NOT say. "
        "If something is uncertain or not mentioned, say that clearly. "
        "Do not invent findings or diagnoses. Keep it calm and supportive.\n\n"
        f"Output language: {language}\n\n"
        "Return JSON with keys: main_findings, suspected_issues, follow_up, summary. "
        "Use arrays for list fields and a short string for summary. "
        "Use plain language that a non-expert can understand.\n\n"
        "MRI report text:\n"
        f"{report_text}"
    )


def call_openai(report_text: str, language: str, model_name: str, temp: float, max_out: int) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    payload = {
        "model": model_name,
        "input": build_prompt(report_text, language),
        "temperature": temp,
        "max_output_tokens": max_out,
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    output_text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    output_text += block.get("text", "")

    if not output_text:
        raise RuntimeError("No response text returned from API.")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return {"raw_output": output_text}


if run:
    if not mri_text.strip():
        st.warning("Paste MRI report text before running.")
    else:
        with st.spinner("Contacting OpenAI..."):
            try:
                status.info("Submitting request...")
                result = call_openai(mri_text, target_language, model, temperature, max_tokens)
                status.success(f"Completed at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as exc:  # noqa: BLE001
                status.error(str(exc))
                result = None

        if result:
            st.subheader("Interpretation")
            st.json(result)
            if "raw_output" in result:
                st.info("The model output was not valid JSON. Showing raw output above.")
