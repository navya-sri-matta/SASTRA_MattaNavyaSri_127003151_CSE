STEP 17 — GRAPH + STRICT CANDIDATE ENTITY LABELS

1. Stop Streamlit with Ctrl+C.
2. Copy app.py, architecture_analysis.py, dependency_graph.py into your existing autosar_hld_assistant project directory.
3. From the project directory, run:
   ..\.venv\Scripts\python.exe -m pip install -r requirements_step17.txt
   ..\.venv\Scripts\python.exe -m streamlit run app.py
   (Copy requirements_step17.txt to your project directory too.)
4. Re-upload AUTOSAR_LayeredSoftwareArchitecture.pdf.
5. Open Dependency graph. Hover over nodes to inspect links.
6. Candidate entities should no longer list 'completely', 'specified', etc.

No other app data needs to be removed. This is a conservative rule-based pilot, not full arbitrary-HLD entity extraction.
