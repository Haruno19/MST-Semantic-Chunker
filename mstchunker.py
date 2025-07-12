from jina_text_segmenter import chunk_text #Gist from https://gist.github.com/MeMartijn/8d66a181f32304de9c07c2529649c35b
from embedding_function import EmbeddingFunction 
from scipy.spatial.distance import cosine
import numpy as np


class MSTChunker():


    def __init__(self):
        self.embedder = EmbeddingFunction()
        return
    

    def split_text(self, text:str) -> list[str]:
        self.chunks = chunk_text(text) # rule-based pre-chunking 

        self.embeds = self.embedder.embed(self.chunks) 
        self.token_lengths = [self.embedder.count_tokens(chunk) for chunk in self.chunks]
        indices = [i for i in range(0, len(self.embeds))]

        connected_components = self._MST_clustering(indices)
        chunks = self._merge_chunks_from_components(connected_components, 400)

        return chunks
    

    def export_chunks_to_md(self, chunks:list, output_path:str, encoding="utf-8"):
        open(output_path, "w", encoding=encoding).close() #clear out file
        with open(output_path, "a", encoding=encoding) as f:
            for i, chunk in enumerate(chunks):
                f.write(f"### Chunk {i} ###\n")
                f.write(f"{chunk}\n\n")
        return True


    ## Clustering with Minimum Spanning Tree (Kruskal)
    def _MST_clustering(self, indices:list[int]) -> list[list[int]]:
        #---tunable parameters
        distance_threshold = 6 # how far apart is too far apart for two chunks to be clustered together if semantically similar
        alpha = 2.26 # empirically chosen to shape lambda's decay curve
        #---------------------

        parent = {}
        rank = {}

        def find(u:int):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u:int, v:int):
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            else:
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
            return True

        def union_cut(u:int, v:int):
            ru, rv = find(u), find(v)
            if ru != rv:
                if rank[ru] < rank[rv]:
                    parent[ru] = rv
                else:
                    parent[rv] = ru
                    if rank[ru] == rank[rv]:
                        rank[ru] += 1


        # Initialize Union-Find for MST
        for idx in indices:
            parent[idx] = idx
            rank[idx] = 0

        # Build edges (distance matrix)
        edges = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u, v = indices[i], indices[j]
                if(abs(u - v) >= distance_threshold):
                    continue  #only add edges that will realistically be chunked together for optimization
                d = self.distance(u, v)
                edges.append((d, u, v))

        distances = [d for d, _, _ in edges]
        print("mean distance: "+str(np.mean(distances)))
        print("mean length: "+str(np.mean(self.token_lengths)))
        alpha = 1.8 * np.mean(distances) + 1.6 # claude ver.
        self.lmbd = np.mean(distances) ** alpha 
        self.lmbd = 0.20

        # Sort edges by weight
        edges.sort()

        # Build MST using Kruskal
        mst_edges = []
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((weight, u, v))
                if len(mst_edges) == len(indices) - 1:
                    break

        # Reset Union-Find to build components after lambda-cut
        parent = {idx: idx for idx in indices}
        rank = {idx: 0 for idx in indices}

        # Apply lambda-cut: union only if weight <= self.lmbd
        for weight, u, v in mst_edges:
            if weight <= self.lmbd:
                union_cut(u, v)

        # Build connected components
        components = {}
        for idx in indices:
            root = find(idx)
            components.setdefault(root, []).append(idx)

        return list(components.values())
    
    ## no max chunk length logic
    # def _merge_chunks_from_components(self, components):
    #     merged = []
    #     for component in components:
    #         merged_chunk = "\n".join(self.chunks[i] for i in component)
    #         merged.append(merged_chunk)
    #     return merged

    def _merge_chunks_from_components(self, components:list[list[int]], max_tokens:int = 400):
        merged = []

        for component in components:
            subchunk = []
            token_sum = 0

            for idx in component:
                chunk = self.chunks[idx]
                chunk_tokens = self.token_lengths[idx]

                # If adding this chunk would exceed max_tokens, truncate current group
                if token_sum + chunk_tokens > max_tokens and subchunk:
                    merged.append("\n".join(subchunk))
                    subchunk = []
                    token_sum = 0

                subchunk.append(chunk)
                token_sum += chunk_tokens

            # Add remaining subchunk
            if subchunk:
                merged.append("\n".join(subchunk))

        return merged
    

    def distance(self, a:int, b:int):
        ## semantic distance
        semantic_distance = cosine(self.embeds[a], self.embeds[b])


        ## positional penalty  
        #---tunable parameters
        gamma = 0.0275 #scale factor for positional penalty
        #---------------------

        sequential_distance = abs(a - b)
        # Penalty: increase distance for far & long chunks
        #penalty = gamma * sequential_distance #linear
        #penalty = np.exp(gamma * sequential_distance) - 1 #non-linear
        penalty = gamma * sequential_distance * np.log(1 + sequential_distance/2) # claude ver.

        ## vicinity reward
        #---tunable parameters
        short_threshold = 80 # 250c, 80t; how short is a "short" chunk (in tokens)
        min_len = 5 # 20c; minimum length of a chunk (in tokens)
        long_term_window = 2 # 2; how far apart chunks are still considered close
        immediate_window = 1 # 1; used for case 2 (the header case)
        vicinity_reward = 0.275 # 0.315; scaling factor of the bullet list item bias
        heading_reward = 0.85 # 0.9; scaling factor of the header bias
        #---------------------

        len_a = max(self.token_lengths[a], min_len) #max(len(self.chunks[a]), min_len)
        len_b = max(self.token_lengths[b], min_len) #max(len(self.chunks[b]), min_len)
        reward = 0

        if sequential_distance <= long_term_window:
            if len_a < short_threshold and len_b < short_threshold:
                reward -= vicinity_reward * np.exp(-min(len_a, len_b) / short_threshold)
        if sequential_distance == immediate_window:
            if len_a < short_threshold:
                reward -= heading_reward * np.exp(-len_a / short_threshold) 


        ## final distance
        #---tunable weights
        semantic_weight = 1.20
        locality_weight = 1.1
        vicinity_weight = 0.735
        #---------------------

        true_dist = semantic_distance*semantic_weight + penalty*locality_weight + reward*vicinity_weight

        return true_dist