STEP 18 — LOCAL PDF REVISION COMPARISON

1. Stop Streamlit (Ctrl+C).
2. Back up your current app.py.
3. Copy these four files into your existing autosar_hld_assistant project folder:
   app.py, architecture_analysis.py, dependency_graph.py, revision_comparison.py
4. Your existing libraries suffice; NO new pip installation is required.
5. From project folder run:
   ..\.venv\Scripts\python.exe -m streamlit run app.py
6. Upload your regular PDF in the main uploader, then open the
   'Compare revisions' tab and upload older/newer versions there.
7. For a basic smoke test, compare two identical-copy PDFs; the app should
   report they are byte-for-byte identical (no changes).

Limitations: comparison uses normalized text lines, and architecture change
flags use the limited AUTOSAR rule catalog. Changes are for review, not approval.
