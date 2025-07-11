from mstchunker import MSTChunker

with open("./docs/sample.md", "r", encoding="utf-8") as f:
    text = f.read()

chunker= MSTChunker()

chunks = chunker.split_text(text)
chunker.export_chunks_to_md(chunks=chunks, output_path="./out/chunked.md")