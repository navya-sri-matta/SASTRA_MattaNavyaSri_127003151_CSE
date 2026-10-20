# AutoArch AI — AUTOSAR HLD Document Analysis Assistant

An educational, local starter implementation for Case Study 1 of the uploaded
Automotive Engineering AI Project Case Studies document. Not a production or
safety-qualified AUTOSAR design-review tool.

## Windows installation (PowerShell)

1. Install Python 3.11 from python.org, ticking **Add Python to PATH**.
2. Open this project folder in VS Code and open a terminal.
3. Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, skip it and use the venv executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Your browser should open http://localhost:8501 .

## Make a sample PDF

Open `sample_hld.txt` in Word or another text editor and **Print to PDF** or
**Save as PDF**. Then upload the PDF in the app.

## Optional local AI answers (Ollama)

1. Install Ollama from https://ollama.com/download .
2. In a terminal, run `ollama pull llama3.2:3b` (requires model download).
3. Ensure Ollama is running, select **Generate a written answer using local Ollama**
   in the app, and click **Search document**.

The base application works without Ollama, API keys or internet access after
Python package installation. It uses TF-IDF retrieval rather than the
case study's recommended embedding + ChromaDB pipeline, to make the first
phase easier to run.

## Features implemented

- PDF text ingestion with page numbers
- Keyword-aware TF-IDF passage retrieval with source pages
- Optional local LLM answers through Ollama; model instructed to cite pages
- Prototype rule-based candidate component/interface/port/signal extraction
- Extracted page viewer and CSV/JSON export

## Important limitations and next milestones

- No OCR for scanned PDFs
- No semantic vector embeddings or ChromaDB yet (planned Phase 2)
- No formal AUTOSAR XML/ARXML parser, dependency graph, revision comparison,
  inconsistency reasoning, or human-approval workflow
- Rules for candidate entities expect labels such as `Component: Foo`
- PDF tables and diagrams may not be interpreted correctly
- LLM citations and claims must be manually verified; no safety approval
- 20 MB maximum PDF file size; data kept only in the running Streamlit session

Only upload documents you have permission to use. Do not upload proprietary
engineering data into unapproved deployments.
