from pathlib import Path
from pypdf import PdfReader

PDF = Path(r"C:\Users\HP\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\9AC3FA3F28F8D3E979A1C8DE94382A58F31525C8\transfers\2026-18\13thBatchDocument.pdf")
reader = PdfReader(str(PDF))
print(f"pages={len(reader.pages)}")
for i, page in enumerate(reader.pages[:12], start=1):
    text = page.extract_text() or ""
    print(f"\n--- PAGE {i} ---")
    print(text[:3000])
