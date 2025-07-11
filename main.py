from mstchunker import MSTChunker

with open("./docs/sample.md", "r", encoding="utf-8") as f:
    text = f.read()

ch = MSTChunker()

chunks = ch.split_text(text)
ch.export_chunks_to_mds(chunks=chunks, output_path="./out/chunked.md")