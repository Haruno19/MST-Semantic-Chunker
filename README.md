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
A `distance_threshold` **tunable** parameter is introduced, and by default set to an **empirically evaluated** value of `6`.  
```python
edges = []
for i in range(len(indices)):
	for j in range(i + 1, len(indices)):
		u, v = indices[i], indices[j]
		if(abs(u - v) >= distance_threshold):
			continue
		d = self.distance(u, v)
		edges.append((d, u, v))
```
  
It’s important to note that the term “*operations count*” is loosely used in the paragraph above. Ultimately, the above code repeats the internal loop exactly as many times as the *pre-optimization* version does. However, the **key difference** is that the `self.distance()` function —which constitutes the major cause of complexity in the sense that it’s the most **computational heavy** step of the algorithm— is only called a total of $(\text{distance-threshold}-1)n$ times in the final optimized version.   
Further optimization could, *in principle*, be achieved by changing the `range` of times the internal loop is repeated to the appropriate value, instead of simply relying on the `if` condition. However, the computational gain granted by skipping the loop’s overhead and `if` evaluation is likely trivial, and becomes a *“demerit”* if we consider the current code’s much better **readability**.   

### Chunk-Merging Problem
Another core (albeit purely “syntactical”) step of the MST-SC algorithm is the transformation of clusters *(connected components)* into the output chunks.  
  
Depending on the specifics of the pre-chunker, input chunks might be **shorter** or **longer**. Anyhow, while the final objective is to find an ultimately consistent balance inherent to the algorithm’s tunable parameters, if not artificially enforced, there’s no **direct control** over the length of the single final output chunk.   
  
Ideally, once the MST clustering computes the connected components, each consisting of a set of **unique** and tight **input** chunks, merging those chunks into a bigger piece of text without any ulterior overhead or logic, would be enough to effectively create the final output chunks. However, to ensure the final chunks don’t “*explode*” in length, a `max_tokens` **tunable** parameter (by default set to `400`) is available as a parameter passed to the `_merge_chunks_from_components()` function.  
This ensures the chunks resulting of MST-SC don’t go over `400` tokens in length. In case combining the input chunks in a connected component exceeds this limit, the the chunk is simply **truncated** and split in two separate output chunks (similarly to the behavior of Chroma’s CSC algorithm under comparable circumstances).   
  
Given this limitation, it’s even more important to use a **quality pre-chunker** that would split the input text in **many**, **short** chunks.   
  
### Experimental Distance Measure  
As mentioned before, the **distance measure** is ultimately the core innovation of this algorithm, and as such, in its current state it’s still **experimental** and likely **unstable**.   
  
The **main goal** with this measure is to incorporate a **semantical metric**, like *cosine similarity*, with a more often used **positional/sequential metric**. 
From a very high-level perspective, this means that the objective of this distance measure is to determine how “*distant*” two chunks of text from an input document are, balancing **how similar in content they are** *(semantical distance)* with **how close or far apart they are within the document** *(positional distance)*.  
  
Abstractly speaking, while not rigorously proven, this *should* improve the consistency and coherence of the chunks that will eventually be possibly retieved by a RAG system and prompted to an LLM as further context for a user query. Nevertheless, I thought the concept was worth a shot in and of itself, and given the *better than expected* immediate results MST-SC demonstrated (by manually and heuristically inspecting the resulting chunks over a handful of different documents), I decided to open source and share this project.  
    
Next, I’d like to dive into the details of how I designed the distance measure to reflect the high-level characteristics described above.   

Please note that, as claimed multiple times, the effectiveness of this metric and algorithm as a whole has been only proven **empirically** and **heuristically** over a handful of documents, and at the moment there’s no underlying mathematical definition or basis “*justifying*” it.   
This is to say, any contribution, suggestion or feedback, and even harsh critique of this chunking approach is highly valuable for the scope of this project!   
  
#### The Overall Metric
Given two text chunks `a` and `b`, the distance between them is defined by the **linear combination** of three factors, each weighed by their respective weight:  
  
$$  
\text{distance}(a,b) = \text{semantic-distance}(a,b)\times w_1 + \text{positional-penalty}(a,b)\times w_2 + \text{vicinity-reward}(a,b)\times w_3  
$$
  
Each of this component —**Semantic Distance**, **Positional Penalty**, and **Vicinity Reward**— has a clearly defined conceptual meaning and motivation, that will be analyzed and explained in the following sections.   
#### Semantic Distance  
The base metric of this distance measure is the **cosine** (dis-)**similarity**, a semantic distance metric.  
```python
from scipy.spatial.distance import cosine
semantic_distance = cosine(self.embeds[a], self.embeds[b])
```  
  
**Scipy**’s implementation of the `cosine` distance is used *as-is* to calculate the semantic distance between chunks `a` and `b`.  
  
As for **why** this metric is used as a base for the entire measure, has mainly to do with the main ambition of this chunker: creating **semantically tight** chunks of the original text to help the retrieval process bring up consistently useful contextual information.  
  
#### Positional Penalty
The concept of a **positional penalty** arises from the need to take into account the sequentiality of chunks (mainly paragraphs) within a document, in relation to how humans use to parse them.   
  
While it’s preferable to have paragraphs focused on the **same topics** clustered together in a single chunk, it often happens that the embedding might not represent with high-enough precision the **surroundings** of the main topic being discussed in a paragraph.   
What this means practically, is that paragraphs that may result in semantically similar embeddings, will likely not bring up **relevantly consistent** information if they’re found at **largely distant positions** within the document.   
Bringing up the same example as before, the *3rd* and *26th* paragraphs of a document should ultimately **not** be chunked together, even if their embeddings are semantically similar.   
  
This **positional penalty** term is therefore designed to make the base semantic distance larger the more the chunks are far apart in terms of position within the document.   
It relies on a scalar and tunable parameter $\gamma$, that pretty much like every other parameter in this code at the moment, has been adjusted empirically, to the value of `0.0275`.  
```python
gamma = 0.0275 #scale factor for positional penalty
```
  
The **sequential distance** of chunks `a` and `b` is calculated with a simple subtraction of their values, which are in fact the **indices** of the input chunks, and conveniently represent their **position** within the document (i.e. `a = 4` means chunk `a` is the 5th chunk in the input document).   
```python
sequential_distance = abs(a - b)
```  
  
The **penalty value** itself is instead computed as a **non-linear function**, scaled by the $\gamma$ parameter mentioned above, to ensure that **the more far apart** the chunks are (i.e. the larger the `sequential_distance` is), **the bigger** the resulting `penalty` is, discouraging the clustering algorithm to put them in the same connected component if their **sequential distance** is relatively high.   
```python
penalty = np.exp(gamma * sequential_distance) - 1
```
  
#### Vicinity Reward  
The third and last component of this custom distance measure is the more **arbitrary** and **biased** one, but for good reasons.  
  
Originally, the distance function was only composed of the **base semantic metric** and the **positional penalty** previously discussed. It was empirically noticed however, that semantically distant but **sequentially close and short** chunks —like *bullet list items* and even *headers* from their *immediately following paragraph*— where very often split apart in many, very short micro-chunks.   
  
This behavior is highly undesirable, as a potential RAG system making use of this chunking algorithm to store its contextual data, will likely end up retrieving this short chunks —like bullet list items or headers— with very high similarity scores to the user’s query, but due to their shortness and inherent lack of relevant information, they will effectively be of very low to null utility in providing the LLM with the further context it needs and expects to retrieve.   
  
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
  
The **length** of each chunk is retrieved from the `self.token_length` list populated in the initialization fase of the algorithm. A “*safety measure*” to avoid length values potentially being set to $0$ (or an excessively small value) is implemented with the `max` function.  
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
  
It’s important to note that, given its *distance shrinking* role, the **vicinity reward**, when applicable, is a **negative** value.  
  
*(Note: it’s important for the second case specifically that chunk `a` comes sequentially before chunk `b`, as the goal is to encourage the clustering algorithm to join the header with its immediately **following** paragraph, and not the header with its immediately **preceding** paragraph. In fact, in both of those cases, the `sequential_distance == immediate_window` condition will be `True`. In this sense, it’s essential to compute the **upper** triangle of the distance matrix as opposed to the bottom one when performing the optimization step).*  
  
#### Weights
After all the three components are computed accordingly to their definitions, the final distance `true_dist` is calculated. As mentioned at the start of this section, each component is scaled by a weight, which too has been determined empirically.    
```python
semantic_weight = 1.20
locality_weight = 1.1
vicinity_weight = 0.735
```  
  
The final calculation is a simple **linear combination** of the three components.  
```python
true_dist = semantic_distance*semantic_weight + 
			penalty*locality_weight + 
			reward*vicinity_weight
```  
  
#### Lambda
Finally, to close this section, I must mention how the $\lambda$  parameter, `self.lmbd`, is defined and computed.   
  
Determining the exact value of *lambda* proves to be perhaps the most quality-impacting operation of the entire algorithm —after all, it’s the *“grind size”* through which the **granularity** of the clustering is determined.   
Initially, an empirically tested value of around `0.47` was considered the sweet-spot for *lambda*, but it’s clear that its entity depends strictly on how intrinsically **tight** the input chunks are with respect to one another. In other words, a **fixed value** for *lambda* would realistically only work (which in this context is to intend as *“produce a chunking of heuristically good quality”*) with documents that once pre-chunked, produce an input chunks set with a **mean distance** within a specific range.   
  
To solve this fundamental issue, an **adaptive calculation** of `self.lmbd` was introduced, based on the **mean distance** of the entire input chunk set, through the application of a **non-linear exponential function** with an empirically set scaling factor.  
```python
alpha = 2.26
...
distances = [d for d, _, _ in edges]
self.lmbd = np.mean(distances) ** alpha
```  
  
This definition of *lambda* is still **tentative** and completely **open to radical revisions**.  
  
## Possible Future Developments
This is just the beginning step of this project. From now one, I’d love to continue making improvements and additions to this algorithm, in order for it to maybe oned day actually become competitive in the chunking algorithms landscape.   
  
I considered evaluating MST-SC with the aforementioned Chroma Evaluating-Chunking suite, but given its (MST-SC’s) intended feature of inherently re-arranging chunks of the original text to form semantically tighter clusters, the standard metrics used for chunking evaluation aren’t really applicable, as the chunks MST-SC produces are not, by design, strict substrings of the original input text.   
Nevertheless, building some sort of evaluation framework in order to gather actually meaningful data on the performance of MST-SC is likely the next step of this project down the line.    
  
Another interesting prospect, given the high amount of tunable parameters which especially the distance function relies on, would be to learn those parameters in a “traditional” machine-learning way. This step however, has the very practical and fundamental challenge that finding quality training data, in this case, text documents paired with their “optimally chunked” version, is. Given the way most current chunking algorithms work, it’s also particularly hard to find well-chunked documents with the original paragraphs re-arranged on the basis of semantic proximity, coherency & consistency, and a dataset not formatted this way, would effectively be useless in the scope of training the MST-SC algorithm. Furthermore, even assuming such a dataset could maybe be generated with the help of a sophisticated LLM, it's still very complicated to formally and expressively quantify the “loss”, that is, the “quality” of an output in comparison to the expected result. 
While very unlikely to happen, given how valuable its results would be, this still remains an open prospect for future developments. 
