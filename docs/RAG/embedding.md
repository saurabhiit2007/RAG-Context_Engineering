### 1. What Are Embeddings in RAG

Embeddings are fixed length vector representations of text that encode semantic meaning.

In a RAG pipeline, embeddings are used to:
- Represent user queries
- Represent documents or chunks stored in a vector index
- Enable similarity based retrieval using cosine similarity, dot product, or Euclidean distance

---

### 2. Dense vs Sparse vs Hybrid Embeddings

### 2.1 Dense Embeddings

Dense embeddings map text into low dimensional continuous vectors, typically 256 to 4096 dimensions.

**Examples**

- Sentence BERT
- E5, GTE, Instructor models
- OpenAI text embedding models

**Pros**

- Capture semantic similarity beyond exact token overlap
- Handle paraphrases and natural language queries well
- Efficient approximate nearest neighbor search

**Cons**

- Weak at exact keyword matching
- Less interpretable
- Sensitive to domain shift

---

### 2.2 Sparse Embeddings

Sparse embeddings represent text as high dimensional sparse vectors aligned with vocabulary terms.

**Examples**

- TF-IDF
- BM25
- SPLADE

**Pros**

- Strong lexical matching
- Interpretable scores
- Robust for rare terms, IDs, and numbers

**Cons**

- Poor semantic generalization
- Vocabulary dependent
- Large memory footprint

---

### 2.3 Hybrid Embeddings

Hybrid approaches combine dense and sparse signals.

**Common strategies**

- **Late fusion of dense and BM25 scores:** In this approach, you run two separate searches—one using semantic vectors (dense) and one using keyword matching (BM25)—and then combine their results using a ranking algorithm like Reciprocal Rank Fusion (RRF). It is called "late" fusion because the merging happens only after both independent retrieval processes are complete.
- **Two stage retrieval with sparse recall and dense reranking:** This strategy uses a fast, keyword-based search (sparse) to quickly narrow down millions of documents to a few hundred candidates, which are then re-scored by a more expensive semantic model (dense). This balances efficiency and accuracy by using the dense model only on a small, pre-filtered subset of data.
- **Joint dense sparse representations:** This involves using a single model or index that generates vectors containing both semantic signals and lexical "importance" weights (like SPLADE). Instead of running two separate searches, you perform one "unified" search that recognizes both the meaning of the sentence and the specific importance of the words within it.

---

### 3. Sentence, Document, and Chunk Level Embeddings

### 3.1 Sentence Level Embeddings

Each sentence is embedded independently.

**Use cases**

- FAQ style retrieval
- Short factual queries

**Limitations**

- Loses broader context
- Sensitive to sentence segmentation errors

---

### 3.2 Document Level Embeddings

Entire documents are embedded as a single vector.

**Use cases**

- Small documents
- Metadata driven retrieval

**Limitations**

- Poor recall for long documents
- Information dilution

---

### 3.3 Chunk Level Embeddings

Documents are split into chunks, often with overlap, and each chunk is embedded.

**Why chunking dominates in RAG**

- Preserves local context
- Improves recall
- Scales to long documents

> Note: Chunking strategy and embedding strategy must be designed together.

---

### 4. Bi Encoders vs Cross Encoders

### 4.1 Bi Encoders

Query and document are encoded independently.

$$
s(q, d) = \langle f(q), g(d) \rangle
$$

where:

- $s(q, d)$: The similarity score between a Query ($q$) and a Document ($d$).
- $f(q)$: The embedding vector of the Query, produced by an encoder model $f$.
- $g(d)$: The embedding vector of the Document, produced by an encoder model $g$ (often the same model as $f$).
- $\langle \dots, \dots \rangle$: This symbol denotes the Dot Product (or inner product) between the two vectors.

This formula highlights the independence of the encoding process. Because $f(q)$ and $g(d)$ are calculated separately, you can pre-calculate $g(d)$ for millions of documents and store them in a database. When a user asks a question, you only need to calculate $f(q)$ once and then perform a fast matrix multiplication to find the best match.

**Pros**

- **Fast retrieval:** Pre-computed document embeddings allow query matching via simple math (dot product) rather than a full model pass.
- **Scales to millions of documents:** Retrieval time grows logarithmically rather than linearly because you only encode the query once.
- **Enables vector indexing:** Compatibility with Approximate Nearest Neighbor (ANN) algorithms allows for high-speed searching across massive datasets.

**Cons**

- **Limited query document interaction:** Because query and document are encoded in isolation, the model cannot perform "cross-attention" to see how specific words in the query relate to specific words in the document.

**Used in**

- **First-stage retrieval:** Acts as a high-speed "filter" to quickly narrow down a massive library of documents to a small candidate set (e.g., the top 100).

---

### 4.2 Cross Encoders

Query and document are encoded jointly.

$$
s(q, d) = h([q; d])
$$

where:

- $[q; d]$: This represents concatenation. The Query and Document are joined together into a single long sequence of text (usually separated by a special token like [SEP]).
- $h(\dots)$: This is the Transformer model. The entire joined sequence is passed through all layers of the model at once.
- $s(q, d)$: The final similarity score is typically the value of the "Classification" head (the [CLS] token) at the very end of the model.

**Pros**

- **Strong relevance modeling:** The model views the query and document simultaneously, allowing it to capture the complex relationship between the user's intent and the content.
- **Fine-grained token interactions:** Every word in the query can directly compare itself to every word in the document via the Transformer's self-attention mechanism.

**Cons**

- **Computationally expensive:** Every query-document pair requires a full forward pass through a deep Transformer, leading to high latency and high GPU costs.
- **Cannot be indexed:** Because the score depends on the specific combination of query and document, you cannot pre-calculate or store results in a vector database for fast lookup.

**Used in**

- **Reranking top-K retrieved chunks:** Cross-Encoders are virtually never used for initial search, it acts as a "high-precision filter" that re-evaluates a small number of candidates (usually 10–100) provided by a faster first-stage retriever.

> Note: Bi encoders maximize recall, cross encoders maximize precision.

### 4.3 Late Interaction (e.g., ColBERT)

The "middle ground" between Bi and Cross encoders.

* **Mechanism:** Stores a vector for *every token* in a document. Retrieval uses a "MaxSim" operation.
* **Benefit:** Achieves Cross-encoder accuracy while remaining much faster for search.

---

### 5. Embedding Training Objectives

Embedding Training Objectives refers to the mathematical strategy used to "teach" a model how to place similar items close together and dissimilar items far apart in a vector space.

### 5.1 Contrastive Learning

Positive pairs are pulled closer while negatives are pushed apart.

$$
\mathcal{L} = -\log \frac{\exp(sim(q, d^+))}{\exp(sim(q, d^+)) + \sum \exp(sim(q, d^-))}
$$

where:

- $\mathcal{L}$: The InfoNCE Loss value. The InfoNCE loss is the negative log-likelihood of the model correctly identifying the single positive document among a set of negatives; minimizing this forces the model to pull relevant pairs together and push irrelevant pairs apart in the vector space."
- $q$: The Query (or Anchor) vector.
- $d^+$: The Positive Document vector (the ground-truth relevant document).
- $d^-$: The Negative Document vectors (irrelevant documents in the batch).
- $sim(u, v)$: The Similarity Function, usually Cosine Similarity or Dot Product, which measures how close two vectors are.
- $\exp(\dots)$: The Exponential Function, used to ensure all similarity scores are positive and to amplify the difference between the highest and lowest scores.
- $\sum$: The Summation over all negative samples in the batch.

> Note: In practice, you will often see a $\tau$ (tau) symbol in this equation: $\exp(sim(q, d) / \tau)$.
> - $\tau$ (Temperature): A hyperparameter that scales the similarity scores.
> - Why it matters: A low temperature makes the model more "opinionated," focusing heavily on the hardest negatives, while a high temperature smooths the distribution.

**Example Losses used**

- **Information Noise-Contrastive Estimation (InfoNCE)**: Described above
- **Multiple negatives ranking (MNR) loss**: MNR Loss is a specific implementation of contrastive learning that is the "bread and butter" of the Sentence-Transformers library.
    - What it means: It is a framework designed for efficient training. Instead of manually finding $100$ "wrong" documents for every question, it uses In-Batch Negatives.
    - How it works: In a batch of $K$ pairs $\{(q_1, d_1), (q_2, d_2), \dots, (q_K, d_K)\}$, the model assumes $d_1$ is the only correct answer for $q_1$. It then treats all other documents in that same batch (7$d_2, d_3, \dots, d_K$) as negative examples for 8$q_1$.
    - Why it's popular: It allows you to train on millions of pairs without ever needing to label "negative" data. You only need $(query, +ve\_document)$ pairs.

---

### 5.2 Supervised Retrieval Objectives

Refers to the final stage of training where you move beyond general semantic similarity and "teach" the model to follow human-labeled preferences for specific queries.
Uses labeled query document relevance data.

**Examples**

- **MS MARCO style datasets:** Named after Microsoft’s MAchine Reading COmprehension dataset, these are the "gold standard" for training RAG retrievers.
   - The Structure: It consists of real-world anonymized Bing queries paired with web passages that were manually marked as "relevant" or "irrelevant" by human judges.
   - Why it matters: Most embedding models (like BGE, GTE, or E5) are fine-tuned on MS MARCO because it teaches the model the "behavior" of a search engine: identifying specific passages that directly answer a natural language question.

- **Pairwise and listwise losses**: Once you have these labeled datasets, you need a mathematical "rule" to update the model. These two methods differ in how many documents they compare at once.
   - **Pairwise Loss (Comparing Two)**
      - The Logic: The model is given a query ($q$), a positive document ($d^+$), and a negative document ($d^-$). The loss function penalizes the model if the score for $d^-$ is higher than (or too close to) the score for $d^+$.
      - Analogy: A "Head-to-Head" tournament. The model only needs to know that A is better than B.
      - Example: RankNet or Triplet Loss.
   - **Listwise Loss (Comparing the Whole List)**
      - The Logic: The model takes a query and a whole list of $N$ documents. It attempts to optimize the entire ranking order simultaneously, rather than just looking at pairs. It is designed to directly improve metrics like NDCG (Normalized Discounted Cumulative Gain).
      - Analogy: A "Leaderboard." The model tries to get the entire top-10 in the correct order.
      - Example: ListNet or LambdaMART.
      - Pros/Cons: Listwise is more accurate for ranking but much more computationally expensive and complex to implement than pairwise.

---

### 5.3 In Batch Negatives

The Concept: For every query-document pair in a training batch, all other documents in that same batch are automatically treated as "negative" examples for that query.

**Why it matters**

- **Efficient:** It provides a massive number of negative samples "for free" without the need to manually label or load additional data from disk.
- **Scales well:** Increasing the batch size mathematically increases the number of distractors per query ($Batch\ Size - 1$), directly sharpening the model's discriminative power.
- **Common in modern embedding training:** It is the standard mechanism for high-performance models (like BGE, E5, and OpenAI) to learn robust representations at a massive scale.

---

### 5.4 Instruction Tuning for Embeddings

Instruction tuning for embeddings involves training models to generate vector representations that dynamically adapt based on natural language task descriptions.

**Example**

- "Represent the question for retrieving relevant passages"

**Benefits**

- **Better zero shot generalization:** Enables models to handle novel retrieval or classification tasks they weren't explicitly trained on by interpreting the provided instruction.
- **Improved alignment across tasks:** Ensures that a single model can produce distinct, context-aware embeddings for the same text depending on whether the goal is clustering, similarity, or domain-specific search.

### 5.5 Matryoshka Representation Learning (MRL)

* **What it is:** Nesting information so that the first $N$ dimensions of a vector contain the most important features.
* **Why it matters:** Allows for **vector truncation**. You can store a 1536-dim vector but only query the first 256 dimensions to save on storage and compute with minimal accuracy loss.

### 5.6 Instruction-Tuned Embeddings

* **Concept:** Models like `Instructor` or `BGE` that take a prefix (e.g., *"Represent this query for retrieving medical research papers"*).
* **Benefit:** Allows a single model to behave differently across specialized tasks (Search vs. Clustering vs. Classification).


---

### 6. Domain Adaptation (The "Cold Start" Problem) for Embeddings

When generic embeddings (OpenAI/Cohere) fail on specialized data (Legal, Medical, Code):

1. **Continued Pre-training:** Run Masked Language Modeling (MLM) on your private corpus.
2. **Fine-tuning (Contrastive Loss):** Use query-document pairs to "pull" relevant items closer.
3. **GPL (Generative Pseudo-Labeling):** Use an LLM to generate synthetic questions for your unlabeled documents, then train the embedding model on these synthetic pairs.
4. **Adapter based tuning**

---

### 7. Distance Metrics and Vector Normalization

### Common Metrics

- **Cosine similarity:** Measures the cosine of the angle between two vectors, focusing on directional orientation rather than magnitude.
- **Dot product:** Calculates the sum of the products of corresponding components, reflecting both the angle and the combined magnitudes of the vectors.
- **Euclidean distance:** Computes the straight-line distance between two points in space, sensitive to absolute coordinate values.

> Note:
> 1. Cosine similarity with L2 normalized vectors is equivalent to dot product.
> Since L2 normalization sets all vector magnitudes to 1, the denominator in the cosine formula ($||A|| \times ||B||$) becomes 1, leaving only the dot product.
> 2. L2 normalization stabilizes similarity scores and improves ANN search behavior. By projecting all vectors onto a unit hypersphere, normalization removes length-based bias and ensures that approximate search algorithms (like HNSW or LSH) focus purely on semantic direction.

---

### 8. Multilingual and Cross Lingual Embeddings

**Goal**
Retrieve documents written in a different language than the query.

**Approaches**
- Joint multilingual training
- Translation based supervision

**Challenges**
- Language imbalance
- Script level differences

---

### 9. Common Failure Modes & Mitigations


| Failure Mode | Root Cause | Mitigation |
| :--- | :--- | :--- |
| **Semantic Drift** | Retrieved chunks are topically related but irrelevant. | Use a Cross-encoder reranker. |
| **Lost in the Middle** | LLM ignores context in long prompts. | Parent-Document Retrieval (retrieve small chunks, provide large context). |
| **Out-of-Vocabulary** | Search for product IDs or rare part numbers. | Implement Hybrid Search (BM25). |
| **Intent Mismatch** | Procedural query retrieves descriptive content. | Use HyDE (Hypothetical Document Embeddings). |

---

### 10. Evaluation of Embeddings in RAG

### Offline Metrics

- Recall at K
- Mean reciprocal rank
- nDCG

---

### End to End Metrics

- Answer correctness
- Faithfulness
- Latency and cost

**Key insight**
Better retrieval does not always lead to better generation without proper prompting and context selection.

---

### 11. High-Frequency Questions

1.  **Q: Why not use an LLM for retrieval directly?**
    * *A:* Context window limits and $O(N^2)$ attention complexity make it impossible to "read" millions of docs per query. Embeddings provide $O(\log N)$ search.
2.  **Q: Does increasing embedding dimensions always improve performance?**
    * *A:* Not necessarily. It can lead to the "Curse of Dimensionality" where distance metrics become less meaningful and latency increases.
3.  **Q: What is HyDE?**
    * *A:* Hypothetical Document Embeddings. You use an LLM to write a "fake" answer to the query, then embed that fake answer to find real documents. This aligns the query more closely with the document's vector space.
4.  **Q: When should you re-index your Vector DB?**
    * *A:* Any time you change the **Embedding Model**. You cannot compare vectors generated by Model A with those from Model B.

- When do dense embeddings outperform BM25
- When should sparse retrieval be preferred
- Why use bi encoders instead of cross encoders for retrieval
- How would you adapt embeddings to a new domain with no labels
- How do you debug poor retrieval in a RAG system

---

### 12. Practical Design Pattern: The "Gold Standard" Pipeline

1.  **Query Expansion:** Use an LLM to rewrite the query or generate a HyDE response.
2.  **Hybrid Retrieval:** Parallel search using Dense (Vector) and Sparse (BM25).
3.  **Reciprocal Rank Fusion (RRF):** Combine scores from different search methods.
4.  **Reranking:** Pass top 50 results through a Cross-encoder.
5.  **Context Selection:** Pass top 5-10 results to the LLM for final generation.