# read_pdf.py
# Step 1 of RAG: prove we can pull text OUT of a PDF file.

import fitz  # this is PyMuPDF — the library we installed. (Its import name is "fitz".)
from pathlib import Path

# Look inside data/papers/ and grab the first PDF we find there.
papers_folder = Path("data/papers")
pdf_files = list(papers_folder.glob("*.pdf"))  # every file ending in .pdf

if not pdf_files:
    print("No PDF found in data/papers/. Put one there and try again.")
    raise SystemExit  # stop cleanly instead of crashing later

pdf_path = pdf_files[0]  # use the first PDF
print(f"Reading: {pdf_path.name}\n")

# Open the PDF. "doc" now represents the whole document.
doc = fitz.open(pdf_path)

print(f"This paper has {len(doc)} pages.\n")

# Pull the text out of every page and join it into one big string.
full_text = ""
for page in doc:
    full_text += page.get_text()  # extract the text from this page

doc.close()

# Report what we got, and show a small sample so we can eyeball it.
print(f"Total characters extracted: {len(full_text)}\n")
print("----- First 500 characters of the paper -----\n")
print(full_text[:500])
print("\n----- (end of sample) -----")