from mstchunker import MSTChunker

# source = "./docs/sample.md"
# source = "./docs/ancient_rome.md"
# source = "./docs/world_war_one.md"
source = "./docs/industrial_revolution.md"
# source = "./docs/scientific_revolution.md"

with open(source, "r", encoding="utf-8") as f:
    text = f.read()

chunker= MSTChunker()

chunks = chunker.split_text(text)
chunker.export_chunks_to_md(chunks=chunks, output_path="./out/chunked.md")
