STEP 19: Generalized HLD extraction upgrade

1. Stop Streamlit (Ctrl+C).
2. Back up your current app.py and other project code.
3. Copy the five .py files in this folder into your existing autosar_hld_assistant project folder; replace existing .py files.
4. Do not delete .venv or chroma_data. No new pip packages are needed.
5. From the existing project folder, run:
   ..\.venv\Scripts\python.exe -m streamlit run app.py
6. Upload HLD_V2.pdf as the main PDF, then open Candidate entities.
   Expected: 3 components, 2 interfaces, 2 signals; 2 signal-flow candidates.
7. In Compare revisions, use HLD_V1.pdf and HLD_V2.pdf.
   New generalized comparison should show additions of WarningController,
   WarningInterface, BrakeWarning and the BrakeController -> WarningController flow.

LIMITATIONS: Explicit labels and simple sender sentences only. This is rule-based.
It does NOT parse arbitrary engineering narratives, diagrams, ARXML, scanned PDFs,
or establish validated runtime wiring. Engineer review remains mandatory.
