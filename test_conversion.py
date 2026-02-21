#!/usr/bin/env python3
"""Quick test: verify backend conversion works. Run from pdf-editor-backend dir."""
import requests
import sys

API = "http://localhost:8001"  # Change if your backend uses different port

def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    print("✓ Backend health OK")

def test_pdf_to_html():
    # Minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000052 00000 n 
0000000101 00000 n 
0000000206 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref
304
%%EOF"""
    r = requests.post(
        f"{API}/to-html",
        files={"file": ("test.pdf", pdf_content, "application/pdf")},
        timeout=120,
    )
    if r.status_code != 200:
        print(f"✗ PDF→HTML failed: {r.status_code}")
        print(r.text[:500] if r.text else "")
        return False
    data = r.json()
    assert "html" in data, "No html in response"
    print("✓ PDF→HTML conversion OK")
    return True

def main():
    print("Testing backend at", API)
    try:
        test_health()
        test_pdf_to_html()
        print("\nAll tests passed. Backend is working.")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to {API}. Is the backend running?")
        print("  Run: uvicorn main:app --reload --port 8001")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
