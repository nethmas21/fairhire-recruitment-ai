from reader_agent import extract_text, extract_pii

path = r"C:\Users\ASUS\Downloads\Nethma_CV_SLIIT.pdf"

text = extract_text(path)
print("----- FIRST 800 CHARACTERS OF EXTRACTED TEXT -----")
print(text[:800])

print("\n----- EXTRACTED PII -----")
print(extract_pii(text))