import pdfplumber

def extractor(filepath: str) -> list[tuple[int,str]]:
    with pdfplumber.open(filepath) as pdf:
        pages_list = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text == None and not text.strip()=="":
                pages_list.append((i+1,text))
        return pages_list



