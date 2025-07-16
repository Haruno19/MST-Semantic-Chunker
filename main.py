from mstchunker import MSTChunker

in_path = "./docs/"
out_path = "./out/"

source = "sample.md"
#source = "ancient_rome.md"
#source = "world_war_one.md"
#source = "industrial_revolution.md"
#source = "scientific_revolution.md"

with open(in_path+source, "r", encoding="utf-8") as f:
    text = f.read()

chunker= MSTChunker()

chunks = chunker.split_text(text)
chunker.export_chunks_to_md(chunks=chunks, output_path=out_path+source)
