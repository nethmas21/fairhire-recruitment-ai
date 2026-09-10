from reader_agent import extract_text, extract_pii, extract_projects_with_llm, extract_projects_section, model

path = r"C:\Users\ASUS\Downloads\Nethma_CV_SLIIT.pdf"  # swap for the actual CV you tested with

print("LLM configured in reader_agent?", model is not None)

text = extract_text(path)
anonymized = text  # rough stand-in; fine for this check

llm_result = extract_projects_with_llm(anonymized)
print("\n----- LLM-EXTRACTED PROJECTS -----")
print(llm_result)

regex_result = extract_projects_section(anonymized)
print("\n----- REGEX-EXTRACTED PROJECTS (fallback) -----")
print(regex_result)