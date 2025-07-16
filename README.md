# MST Semantic Chunker

A new experimental text chunking method based on **Minimum Spanning Tree** clustering with a **hybrid semantical-positional** distance measure.

---
Inspired by Chroma’s research on [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking), specifically the [Cluster Semantic Chunker](https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py) algorithm proposed and analyzed in the article, I designed from the ground up another **clustering-based** chunking algorithm that would tackle the main challenges of chunking text for information retrieval.  

The **Cluster Semantic Chunker** (CSC) algorithm makes use of dynamic programming to compute clusters of small text chunks, pre-processed with a rule-based chunker, aiming at maximizing the sum of the **cosine similarities** within those clusters, therefore incorporating a **semantic** measure of distance in the definition of the chunks.  
Inspired by this idea, I implemented a similar chunking method relying on a different, peculiar clustering algorithm I had previously explored under [different circumstances](https://github.com/Haruno19/scikit-cautious): the **Minimum Spanning Tree clustering algorithm.** 

*As an initial disclaimer, for reasons that will be more thoroughly explained further on, this algorithm hasn’t yet been benchmarked, aside from simple heuristic and manual/GPT-a-judge evaluations. Building a proper evaluation framework compatible with the features and functionality of this chunking algorithm is the next step down the line!*  

## The General Idea
*How does MST clustering work?*   
In short, nodes (in this case, the pre-chunked short pieces of text) are arranged into a **complete**, **weighted** and **undirected graph**, over which the **Minimum Spanning Tree** $T$ is computed using Kruskal’s method. The MST is than pruned with respect to a parameter $\lambda$, basically creating $n$ **tightly connected components**, each representing a cluster (and ultimately a longer chunk of the original text).  
  
The flow of the **MST Semantic Chunker** (MST-SC) algorithm can be visually summed up as follows:  
<p align="center">
<img width="85%" alt="MST-SC" src="https://github.com/user-attachments/assets/cfff0f3c-a38a-4dec-b145-5be217cb1e4b" />
</p>
The main innovation the MST-SC algorithm aims to introduce however, is its unique **distance measure** used to calculate the weights of the graph’s edges.   
  
The distance measure is the “core” of any clustering algorithm, arguably even more fundamental than the structure of the algorithm itself.   
Measuring distance between chunks of text is notoriously difficult, as there’s no *quick and easy* definition that’s universally accepted or qualifiable as true. Especially in the context of **RAG** systems, when splitting text into smaller chunks, it’s beneficial to consider not only the surface-level syntax and textual delimiters, but also integrating a semantic measure to ensure semantically close paragraphs aren’t split apart, boosting the context retrieval’s performance and effectiveness.   
  
Integrating two such metrics into a single cohesive and consistent measure can be tricky, and Chroma’s CSC algorithm achieves pretty good results with its approach.
In the pursuit of achieving similar capabilities, MST-SC implements an **experimental and tunable distance measure** that incorporates **semantic distance** as well as **positional bias** into one single distance function.  

### Version 2 (Claude ver.)
Leveraging the help of Anthropic's **Claude**, the first major version upgrade of this algorithm came about, featuring much better adaptability over a much wider variety of documents, introducing an automatic tuning process to dynamycally adjust the way the distance measure is calculated based on the input's intrisic characteristics.
  
*(Refer to the corresponding paragraph in the “**In Details…**” section for a more specific and in depth overview of how this measure is defined and computed)*  

## Installation & Usage
To install and test MST-SC’s chunking performance on your own documents, simply `clone` this repository, install its dependencies, and run the `main.py` script (making sure it’s pointing to the correct input file).   
  
The dependencies of MST-SC are listed in the `requirements.txt` file:  
```
sentence_transformers
transformers
scipy
```
It’s also necessary to download the `jina_text_segmenter.py` open source script by [Martijn Schouten](https://gist.github.com/MeMartijn/8d66a181f32304de9c07c2529649c35b#file-jina_text_segmenter-py-L70) and place it in the root folder of the repo (or alternatively, build your own rule-based pre-chunker).   
  
Out of the box, the embedding model used by MST-SC is [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), but it can be changed to your preferred model by changing the `EmbeddingFunction` class in `embedding_function.py` accordingly.  
  
Included in this repo, there’s also a demo markdown input file and its corresponding chunked output. The original document is a collection of (likely poorly-written) notes I recovered from my old high-school folder, parsed into `.md` from a `.docx` file with *Microsoft*’s [markitdown](https://github.com/microsoft/markitdown), which I found surprisingly handy and fitting for this use case.   
Other documents I tested the MST-SC algorithm with (and used as an empirical basis to manually adjust the tunable parameters) include other old notes of mine on different subjects, written in Italian like the demo sample, as well as a handful of *ad-hoc* files written in English by *ChatGPT-4o*.  
  
## In Details…  
  
### Rule-Based Pre-Chunking
As stated above, the rule-based pre-chunking operation is handled entirely by the [Jina AI Text Segmenter](https://gist.github.com/MeMartijn/8d66a181f32304de9c07c2529649c35b) by Martijn Schouten.   
  
MST-SC works best which **highly separated short chunks** as its input, therefore the main objective of this step is to split the input text into **many short chunks** to then feed the clustering algorithm. I found Martijn Schouten’s segmenter to empirically perform the best, but if you want to test the chunking algorithm with another pre-chunker (or you own, custom-built one), I suggest you evaluate its output first and compare it with *Jina Segmenter*’s, to ensure the degree of separation it achieves is comparable.  

### MST Clustering
The process clustering the pre-chunked input text into larger, semantically and locally tight final chunks through MST ordering and pruning, relies pretty much entirely on standard procedures.   
  
To avoid repeating multiple times throughout the algorithm costly operations like chunk embedding, MST-SC relies on a few internal structures, over which it performs its computations.  
- `self.chunks` is of type `list[str]`, and contains the pre-processed input chunks.  
- `self.embeds` is a list of the same length of `self.chunks`, and stores for each chunk its embedding at the corresponding index.  
- `self.token_lengths` analogously store the token length of each chunk at its corresponding index.   
- `indices` is a “symbolic” list, used mainly for clarity of notation, over which the MST and the related operations are computed. As the name suggests, each of its elements is a simple `int` index, that represents the corresponding chunk in `self.chunks` or `self.embeds`, or `self.token_lengths`.  
  
Upon computing the distance matrix, which in itself describes the entire graph, the standard **Kruskal’s Minimum Spanning Tree** method is computer over the graph, with custom `union` and `find` functions (whereas the `union_cut` function is specially implemented to derive the connected components at the end).   
  
Of the resulting tree, only the edges whose weight is **smaller or equal** than the `self.lmbd` $\lambda$ parameter are joined when creating the connected components that will later be treated as the output chunks. The intuition is that cutting the edges that connect nodes *(input chunks)* whose distance is **higher** than the *lambda* threshold, will result in multiple **connected components** where the average **intra-node distance** is arbitrarily low, meaning the components are **tightly** connected, or *“fit well together”* (in respect to the distance metric).  

### Distance Matrix Optimization
On a more practical note, the major drawback of this chunking (and clustering) algorithm lies in the computation of the **pairwise distance matrix**, the matrix which contains the distances from each node to the others, basically defining the input weighted graph.   
```python
edges = []
for i in range(len(indices)):
	for j in range(len(indices)):
		u, v = indices[i], indices[j]
		d = self.distance(u, v)
		edges.append((d, u, v))
``` 
   
A very intuitive and straightforward initial optimization for this process is to avoid calculating redundant and useless distances. By nature, the distance matrix over an **undirected**, **weighted** and **complete graph** (like this one) is **symmetric**, and null (i.e. equal to $0$) over the **main diagonal**.   
  
By computing only the **upper triangle** of the matrix (as opposed to the lower triangle, which would break the sequentiality of the input chunks, a fundamental property in the computation of the distance measure as we’ll see in the sections below), while still achieving an asymptotic complexity of $O(n^2)$, we’re able to cut down the operations count to $\frac{n(n-1)}{2}$, effectively reducing it by half.   
```python
edges = []
for i in range(len(indices)):
	for j in range(i + 1, len(indices)):
		u, v = indices[i], indices[j]
		d = self.distance(u, v)
		edges.append((d, u, v))
```
  
Another non-trivial optimization we can implement is actually dictated by the semantics of the distance function. Further details on this concept will be discussed later on, but in a nutshell, nodes *(chunks)* that are positionally too far apart from each other (say, the *3rd* and the *26th* paragraphs of the input document), will ultimately **not** going to be clustered together, in order to (loosely) adhere to a locality and sequentiality principle.   

What this means effectively, is that the exact value of their distance is inconsequential to core steps of the algorithm (as it’ll turn out to be **significantly higher** than the *lambda* threshold anyways). Therefore, skipping the distance calculation of the values that are arbitrarily too far apart, significantly lowers the operations count.   
  
#### Adaptive Distance Threshold
Version 2 introduces an **adaptive distance threshold** that scales with document characteristics. Instead of a fixed threshold, the algorithm now calculates:
```python
base_threshold = 5
scaling_factor = 0.03
distance_threshold = max(base_threshold, int(len(self.chunks) * scaling_factor))
```

This means that for **longer documents** with more chunks, the algorithm will consider a wider range of chunk pairs for potential clustering, while shorter documents maintain tighter positional constraints. This adaptive approach helps balance **computational efficiency** with **clustering quality** across documents of varying lengths.

### Document Characteristics Analysis
Version 2 introduces a **pre-computation step** that analyzes the document's intrinsic characteristics to adapt the distance function parameters accordingly. This analysis computes two key metrics, the **semantic density** and the **structural fragmentation** of the document.

#### Semantic Density
The **semantic density** measures how semantically coherent and consistenr the document is overall:
```python
self.semantic_density = 1 - np.mean(similarities_for_density)
```
A **higher semantic density** indicates that chunks within the document are generally more semantically similar to each other, suggesting a focused, coherent document. A **lower semantic density** indicates more diverse content with chunks covering disparate topics.

#### Structural Fragmentation
The **fragmentation** metric measures how broken up the document structure is:
```python
short_chunks = sum(1 for length in self.token_lengths if length < np.mean(self.token_lengths)*1.55)
self.fragmentation = short_chunks / len(self.chunks)
```
**Higher fragmentation** indicates many short chunks (headers, bullet points, brief paragraphs), while **lower fragmentation** suggests more uniform, substantial chunks.

### Chunk-Merging Problem
Another core (albeit purely “syntactical”) step of the MST-SC algorithm is the transformation of clusters *(connected components)* into the output chunks.  
  
Depending on the specifics of the pre-chunker, input chunks might be **shorter** or **longer**. Anyhow, while the final objective is to find an ultimately consistent balance inherent to the algorithm’s tunable parameters, if not artificially enforced, there’s no **direct control** over the length of the single final output chunk.   
  
Ideally, once the MST clustering computes the connected components, each consisting of a set of **unique** and tight **input** chunks, merging those chunks into a bigger piece of text without any ulterior overhead or logic, would be enough to effectively create the final output chunks. However, to ensure the final chunks don’t “*explode*” in length, a `max_tokens` **tunable** parameter (by default set to `400`) is available as a parameter passed to the `_merge_chunks_from_components()` function.  
This ensures the chunks resulting of MST-SC don’t go over `400` tokens in length. In case combining the input chunks in a connected component exceeds this limit, the the chunk is simply **truncated** and split in two separate output chunks (similarly to the behavior of Chroma’s CSC algorithm under comparable circumstances).   
  
Given this limitation, it’s even more important to use a **quality pre-chunker** that would split the input text in **many**, **short** chunks.   
  
### Experimental Distance Measure  
As mentioned before, the **distance measure** is ultimately the core innovation of this algorithm. As of Version 2, the stability and adaptability of the distance function has increased significantly, producing much better results over a relatively varied set of documents.
  
The **main goal** with this measure is to incorporate a **semantical metric**, like *cosine similarity*, with a more often used **positional/sequential metric**. 
From a very high-level perspective, this means that the objective of this distance measure is to determine how “*distant*” two chunks of text from an input document are, balancing **how similar in content they are** *(semantical distance)* with **how close or far apart they are within the document** *(positional distance)*. 
  
Abstractly speaking, while not rigorously proven yet, this *should* improve the consistency and coherence of the chunks that will eventually be possibly retieved by a RAG system and prompted to an LLM as further context for a user query. 
    
Next, I'd like to dive into the details of how I designed the distance measure to reflect the high-level characteristics described above.   
  
#### The Overall Metric
Given two text chunks `a` and `b`, the distance between them is defined by the **linear combination** of three factors, each weighed by their respective weight:  
  
$$  
\text{distance}(a,b) = \text{semantic-distance}(a,b)\times w_1 + \text{positional-penalty}(a,b)\times w_2 + \text{vicinity-reward}(a,b)\times w_3  
$$
  
Each of this component —**Semantic Distance**, **Positional Penalty**, and **Vicinity Reward**— has a clearly defined conceptual meaning and motivation, that will be analyzed and explained in the following sections.   

#### Semantic Distance  
The base metric of this distance measure is the **cosine** (dis-)**similarity**, a semantic distance metric. In Version 2, this value is **pre-computed** and cached for efficiency:
```python
# Pre-computed and stored in self.cosine_matrix during initialization
semantic_distance = self.cosine_matrix[(a,b)]
```  
  
As for **why** this metric is used as a base for the entire measure, has mainly to do with the main ambition of this chunker: creating **semantically tight** chunks of the original text to help the retrieval process bring up consistently useful contextual information.  
  
#### Positional Penalty
The concept of a **positional penalty** arises from the need to take into account the sequentiality of chunks (mainly paragraphs) within a document, in relation to how humans use to parse them.   
  
While it’s preferable to have paragraphs focused on the **same topics** clustered together in a single chunk, it often happens that the embedding might not represent with high-enough precision the **surroundings** of the main topic being discussed in a paragraph.   
What this means practically, is that paragraphs that may result in semantically similar embeddings, will likely not bring up **relevantly consistent** information if they’re found at **largely distant positions** within the document.   

This **positional penalty** term is therefore designed to make the base semantic distance larger the more the chunks are far apart in terms of position within the document.   
It relies on a scalar and tunable parameter $\gamma$, that pretty much like every other parameter in this code at the moment, has been adjusted empirically, to the value of `0.0275`.  
```python
gamma = 0.0275 #scale factor for positional penalty
```
  
The **sequential distance** of chunks `a` and `b` is calculated with a simple subtraction of their values, which are in fact the **indices** of the input chunks, and conveniently represent their **position** within the document (i.e. `a = 4` means chunk `a` is the 5th chunk in the input document).   
```python
sequential_distance = abs(a - b)
```  
  
Version 2 introduces a **refined positional penalty calculation** that provides more nuanced control over how distance affects clustering:
```python
penalty = gamma * sequential_distance * np.log(1 + sequential_distance/2)
```

This new formulation combines **linear** and **logarithmic** components:
- The **linear term** (`gamma * sequential_distance`) ensures that penalty increases proportionally with distance
- The **logarithmic term** (`np.log(1 + sequential_distance/2)`) provides a **gentle acceleration** that grows more slowly for very distant chunks

**Compared to the original exponential function**, this new approach:
- **Starts more gently**: For nearby chunks (distance 1-3), the penalty is much smaller than the exponential version
- **Grows more predictably**: The logarithmic component prevents the penalty from exploding for moderately distant chunks
- **Maintains separation**: Still effectively prevents very distant chunks from being clustered together
- **Better balance**: Allows the semantic component to have more influence in medium-distance decisions

This results in **more nuanced clustering decisions** where moderately distant but semantically related chunks can still be considered for clustering, while maintaining the principle that very distant chunks should not be grouped together.

#### Vicinity Reward  
The third and last component of this custom distance measure is the more **arbitrary** and **biased** one, but for good reasons.  
  
Originally, the distance function was only composed of the **base semantic metric** and the **positional penalty** previously discussed. It was empirically noticed however, that semantically distant but **sequentially close and short** chunks —like *bullet list items* and even *headers* from their *immediately following paragraph*— where very often split apart in many, very short micro-chunks.   
  
This behavior is highly undesirable, as a potential RAG system making use of this chunking algorithm to store its contextual data, will likely end up retrieving this short chunks —like bullet list items or headers— with very high similarity scores to the user's query, but due to their shortness and inherent lack of relevant information, they will effectively be of very low to null utility in providing the LLM with the further context it needs and expects to retrieve.   
  
Thus, the idea of a **vicinity reward** was introduced, to **artificially encourage** the clustering algorithm to put in the same connected component, regardless of their semantic distance, the two chunks `a` and `b`, when the following specific two cases occur:  
1. Chunks `a` and `b` are arbitrarily **short** in terms of tokens, and arbitrarily **sequentially close** together'; the *bullet list case*.   
2. Chunk `a` is arbitrarily **short** and `b` follows immediately after `a`; the *header—paragraph* case.  
  
Given the **arbitrariness** of these two situations, the computation of the **vicinity reward** relies on many **tunable parameters**, all currently empirically adjusted as follows:  
```python
min_len = 5 # minimum length of a chunk (in tokens)
short_threshold = 80 # how short is a "short" chunk (in tokens)
long_term_window = 2 # how far apart chunks are still considered close
immediate_window = 1 # used for case 2 (the header case)
vicinity_reward = 0.275 # scaling factor of the bullet list bias
heading_reward = 0.85 # scaling factor of the header-paragraph bias
```
  
The **length** of each chunk is retrieved from the `self.token_length` list populated in the initialization fase of the algorithm. A "*safety measure*" to avoid length values potentially being set to $0$ (or an excessively small value) is implemented with the `max` function.  
```python
len_a = max(self.token_lengths[a], min_len) 
len_b = max(self.token_lengths[b], min_len)

reward = 0 # reward by default is 0, if neither of the two cases occur
```  
  
The two cases are then handled **sequentially**, starting with case 1. Both conditions can be `True` at once; in this case, the reward granted to the *header — paragraph* case is even **larger**.   
Analogously to the positional penalty calculation, the **vicinity reward** too is described through a **non-linear function**.   
```python
#case 1
if sequential_distance <= long_term_window:
	if len_a < short_threshold and len_b < short_threshold:
		reward -= vicinity_reward * np.exp(-min(len_a, len_b) 
				  / short_threshold)
#case 2
if sequential_distance == immediate_window:
	if len_a < short_threshold:
		reward -= heading_reward * np.exp(-len_a / short_threshold)
```  
  
It's important to note that, given its *distance shrinking* role, the **vicinity reward**, when applicable, is a **negative** value.  
  
*(Note: it's important for the second case specifically that chunk `a` comes sequentially before chunk `b`, as the goal is to encourage the clustering algorithm to join the header with its immediately **following** paragraph, and not the header with its immediately **preceding** paragraph. In fact, in both of those cases, the `sequential_distance == immediate_window` condition will be `True`. In this sense, it's essential to compute the **upper** triangle of the distance matrix as opposed to the bottom one when performing the optimization step).*  
  
#### Adaptive Weights
Version 2 introduces **dynamic weight calculation** based on the document's characteristics, replacing the fixed weights of Version 1. The weights are now calculated as:

```python
self.semantic_weight = 0.85 + (self.semantic_density * 0.75)
self.locality_weight = 0.20 + np.exp(-0.25 * self.semantic_density - 0.5 * self.fragmentation)
self.vicinity_weight = 0.125 + 0.60 * np.exp((self.fragmentation - 0.8) * 0.85)
```

**Semantic Weight** adaptation:
- **Higher semantic density** → **Higher semantic weight**: When the document is more semantically coherent, the semantic distance becomes more reliable and influential
- **Lower semantic density** → **Lower semantic weight**: When content is diverse, semantic similarity becomes less trustworthy as a clustering criterion

**Locality Weight** adaptation:
- **Higher semantic density** → **Lower locality weight**: When chunks are semantically similar, positional constraints can be relaxed
- **Higher fragmentation** → **Lower locality weight**: When the document is fragmented, strict positional penalties would prevent beneficial clustering of related fragments
- **Lower semantic density + Lower fragmentation** → **Higher locality weight**: When content is diverse but well-structured, positional information becomes more important

**Vicinity Weight** adaptation:
- **Higher fragmentation** → **Higher vicinity weight**: Documents with many short chunks benefit more from vicinity rewards to group headers with content
- **Lower fragmentation** → **Lower vicinity weight**: Documents with uniform chunk sizes need less artificial clustering encouragement

This adaptive weighting system allows the algorithm to **automatically adjust** its behavior based on the document's structure and content characteristics, leading to more appropriate chunking decisions across different document types.

The final calculation uses these dynamic weights:
```python
true_dist = (semantic_distance * self.semantic_weight + 
             penalty * self.locality_weight + 
             reward * self.vicinity_weight)
```

The many hard-coded values in these calculations are the result of extensive testing over a suite of documents with different characteristics; the values were empirically adjusted to produces the best results over all the documents.

#### Lambda - Adaptive Threshold Calculation
Version 2 significantly improves the lambda calculation to be more adaptive to different document types and characteristics.

The lambda parameter, `self.lmbd`, determines the **granularity** of the clustering by setting the threshold for which edges in the minimum spanning tree are retained. A **smaller lambda** results in **finer granularity** (more, smaller chunks), while a **larger lambda** results in **coarser granularity** (fewer, larger chunks).

The new adaptive calculation is:
```python
distances = [d for d, _, _ in edges]
alpha = 1.8 * np.mean(distances) + 0.85
self.lmbd = np.mean(distances) ** alpha * np.exp(-0.15 / np.mean(distances))
```

**Alpha Calculation**:
The alpha parameter is now **dynamically computed** based on the mean distance of the document:
- **Higher mean distances** → **Higher alpha**: When chunks are generally more distant from each other, a higher alpha creates a more aggressive exponential curve
- **Lower mean distances** → **Lower alpha**: When chunks are generally closer, a gentler exponential curve is used

**Lambda Calculation Components**:
1. **Base component**: `np.mean(distances) ** alpha` - This creates the primary exponential relationship
2. **Correction factor**: `np.exp(-0.15 / np.mean(distances))` - This exponential correction provides:
   - **Gentle reduction** for documents with higher mean distances
   - **Minimal impact** for documents with lower mean distances

**Behavior across document types**:
- **Semantically tight documents** (low mean distances): Lambda is calculated more conservatively to avoid over-clustering
- **Semantically diverse documents** (high mean distances): Lambda is calculated more aggressively to ensure meaningful clusters form despite higher distances
- **Fragmented documents**: The correction factor helps balance the competing needs of clustering related fragments while maintaining appropriate granularity
  
## Possible Future Developments
Version 2 features a significantly stronger adaptability across a much wider variety of documents. This addresses one of the main drawbacks from the original version, making MST-SC remarkably more effective for a real-world use case.  
  
The next big step forward would therefore be a proper evaluation of its capabilies, in comparison to other currently widespread chunking strategies. As it was the case for Version 1, current evaluation techiques and suites aren't able to meaningfully evaluate MST-SC due to its intrisic feature of producing chunks that aren't strictly sub-strings of the original documents, but instead re-arranged pieces to form semantically tight chunks.  
Building a strucutred, effective and un-biased evaluation framework will be the next major step in this research. 