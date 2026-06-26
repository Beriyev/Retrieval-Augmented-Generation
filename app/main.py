from agent import extractor, chunker

pages = extractor("C:/RAG Project/docs/Unit_1.pdf")
chunks = chunker(pages, filepath="C:/RAG Project/docs/Unit_1.pdf")

print(f"Total chunks: {len(chunks)}")
for c in chunks[:5]:  # just look at the first 5
    print(c)