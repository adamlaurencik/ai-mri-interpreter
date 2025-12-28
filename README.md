# MRI Interpretation Spike (Streamlit)

This is a minimal Streamlit spike that accepts MRI report text and returns a structured interpretation using the OpenAI API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
cp .env.example .env
# then edit .env to add your key
```

Set an app password:

```bash
# add APP_PASSWORD to your environment or .env
```

Set an OCR token (only needed for image input):

```bash
# add EDENAI_API_TOKEN to your environment or .env
```

## Run

```bash
streamlit run app.py
```

## Notes

- This is a prototype for rapid iteration, not a diagnostic tool.
- Use de-identified MRI text only.
- Output is a JSON structure with findings, suspected issues, follow-up notes, and a summary.
