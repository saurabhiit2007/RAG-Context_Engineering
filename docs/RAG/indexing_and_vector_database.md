## Indexing Strategies, Vector Databases, and Retrieval Systems for RAG

### 1. Why Indexing and Retrieval Matter in RAG

In RAG systems, retrieval is the bottleneck for answer quality. If the retriever fails, even the strongest LLM cannot recover.

Indexing and retrieval determine:
- What information is even visible to the generator
- Latency and cost per query
- Scalability to millions or billions of chunks
- Robustness to noisy or ambiguous queries

A typical RAG retrieval pipeline:
1. Chunk documents
2. Generate embeddings
3. Build indexes
4. Retrieve candidates
5. Re rank or filter
6. Construct final context

---

## 2. Indexing Strategies

Indexing defines how embeddings or tokens are organized to support fast similarity search.

### 2.1 Flat Index

A flat index stores all vectors and computes similarity against every vector during search.

#### Pros

- **Exact results**: No approximation is used, so recall is perfect. This makes it a reliable reference for evaluating other indexes.
- **Simple implementation**: No clustering, graph construction, or parameter tuning is required.
- **Deterministic behavior**: Results are stable across runs, which simplifies debugging.

#### Cons

- **Does not scale**: Search time grows linearly with dataset size, making it unusable beyond small corpora.
- **High latency**: Even moderate sized datasets cause unacceptable response times.
- **High compute cost**: Requires full distance computation for every query.

#### When to use

- Small datasets
- Offline evaluation
- Ground truth recall benchmarking

---

### 2.2 Approximate Nearest Neighbor Indexes

Approximate methods trade small accuracy loss for large performance gains.

---

### 2.2.1 HNSW (Hierarchical Navigable Small World)

HNSW builds a multi layer graph where each node connects to similar vectors. Search starts at higher layers and progressively refines.

#### Pros

- **Very high recall at low latency**: HNSW often achieves near flat index recall with orders of magnitude faster search.
- **Fast query performance**: Graph traversal limits the number of distance computations.
- **Supports dynamic updates**: New vectors can be inserted without rebuilding the index.

#### Cons

- **High memory usage**: Graph edges add significant overhead compared to raw vectors.
- **Slow index construction**: Building the graph is expensive for very large datasets.
- **Parameter sensitivity**: Poor tuning can increase memory usage or degrade recall.

### Best for

- Latency sensitive RAG systems
- Medium to large scale corpora
- Online and frequently updated data

---

### 2.2.2 IVF (Inverted File Index)

IVF clusters vectors into centroids and searches only within the closest clusters.

> Note:
    In traditional text retrieval, an inverted index maps:
    (term → list of documents containing that term).
    This is the opposite of storing documents as sequences of terms, hence the word inverted. At query time, only documents associated with the query terms are examined.

    In IVF, the mapping becomes:
    - centroid ID → list of vectors assigned to that centroid
    Instead of scanning all vectors, the system:
    1. Assigns each vector to its nearest centroid during indexing
    2. At query time, finds the closest centroids
    3. Searches only the vectors stored in those centroid “lists”
    Each centroid acts like a term in a classical inverted index, and each posting list contains the vectors that belong to that region of the vector space.

### Pros

- **Lower memory overhead**: Does not store large graphs, making it more memory efficient.
- **Faster index build**: Clustering is cheaper than graph construction.
- **Works well with disk based search**: Suitable for large datasets that do not fit fully in memory.

### Cons

- **Lower recall than HNSW**: If relevant vectors fall outside probed clusters, they are missed.
- **Sensitive to clustering quality**: Poor centroids lead to degraded retrieval quality.
- **Requires careful tuning**: Number of clusters and probes strongly affect performance.

### Best for

- Very large scale datasets
- Cost constrained systems
- Disk backed vector stores

---

### 2.2.3 Product Quantization (PQ)

PQ compresses vectors into compact codes and computes approximate distances.

#### Pros

- **Massive memory reduction**: Enables storage of billions of vectors.
- **Lower IO cost**: Smaller representations reduce disk access.

#### Cons

- **Lossy compression**: Precision drops due to quantization errors.
- **Reduced recall**: Fine grained similarity distinctions are lost.
- **Harder to debug**: Errors are harder to attribute to specific vectors.

### Typically used with

- IVF + PQ for extreme scale search

---

### 2.3 Sparse Indexes

Sparse indexes represent documents using term based features such as TF IDF or BM25. Each document is a high dimensional vector over a vocabulary, where most dimensions are zero.

They are implemented using an inverted index, which maps:

term → list of documents containing that term

At query time, only documents that share terms with the query are considered, and they are scored using functions like BM25 that account for term frequency, document frequency, and document length.

#### Pros

- **Strong lexical precision**: Exact matches for keywords, IDs, and entities.
- **Interpretable scoring**: Scores can be explained via term frequency and document frequency.
- **Handles rare terms well**: Especially important for names, error codes, and identifiers.

#### Cons

- **Weak semantic understanding**: Cannot match paraphrases or conceptual similarity.
- **Vocabulary dependent**: Performance degrades when query wording differs from documents.

### Common systems

- Elasticsearch
- OpenSearch
- Lucene

---

### 2.4 Hybrid Indexing

Hybrid systems combine dense and sparse indexes to exploit complementary strengths.

### Pros

- **Improved recall and precision**: Dense captures semantics, sparse captures exact matches.
- **Robust to query variation**: Handles both natural language and keyword heavy queries.
- **Production proven**: Widely used in enterprise and legal search.

### Cons

- **Increased system complexity**: Requires managing multiple indexes and score fusion.
- **Higher latency**: Multiple retrieval paths increase query cost.
- **Tuning complexity**: Weighting dense vs sparse scores is non trivial.

---

### 3. Vector Databases

Vector databases manage embedding storage, indexing, and retrieval at scale.

### 3.1 Core Capabilities

A production ready vector database supports:

- Approximate nearest neighbor search
- Metadata filtering
- Index persistence
- Horizontal scaling
- Online updates
- Observability and monitoring

---

### 3.2 Popular Vector Databases

### FAISS

**Pros**

- Extremely flexible
- High performance
- Ideal for research and custom pipelines

**Cons**

- Not a full database
- Requires significant engineering for production use

---

### Milvus

**Pros**

- Distributed and scalable
- Supports multiple index types
- Strong ecosystem

**Cons**

- Operational complexity
- Requires careful resource management

---

### Qdrant

**Pros**

- Strong metadata filtering
- Simple operational model
- Optimized for RAG workloads

**Cons**

- Less flexible index customization
- Smaller ecosystem compared to FAISS

---

### Pinecone

**Pros**

- Fully managed
- Minimal operational overhead
- Consistent performance

**Cons**

- Less control over internals
- Cost can scale quickly

---

## 3.3 Metadata Filtering

Metadata filtering restricts retrieval to relevant subsets.

### Pros

- Improves precision
- Enforces access control
- Reduces irrelevant context

### Cons

- Over filtering can reduce recall
- Poor filter design can hide relevant documents

Filtering can be applied before or after vector search, each with different tradeoffs.

---

### 4. Retrieval Strategies

### 4.1 Dense Retrieval

Dense retrieval embeds queries and documents into low-dimensional dense vectors using neural encoders.

**Common models:**

- Sentence Transformers
- Contriever
- E5
- GTR

Retrieval is performed using approximate nearest neighbor search.

#### Pros

- Captures semantic similarity
- Robust to paraphrasing
- Domain adaptable via fine tuning

#### Cons

- Weak for exact values
- Sensitive to embedding quality
- Harder to interpret scores

---

### 4.2 Sparse Retrieval

Sparse retrieval represents documents and queries using high-dimensional sparse vectors based on term statistics.

#### Common methods:

- TF-IDF
- BM25
- Inverted Indexes

#### Pros

- Excellent for exact matches
- Strong for structured identifiers
- Interpretable results

#### Cons

- Poor semantic recall
- Fails on natural language variation

---

## 4.3 Hybrid Retrieval

Hybrid retrieval combines sparse and dense retrieval signals.

**Common strategies:**

- Score fusion
- Result union followed by reranking
- Weighted linear combination

#### Pros

- Best overall retrieval quality
- Handles diverse query types
- Reduces failure modes of single retrievers

#### Cons

- Higher system complexity
- Increased latency and cost

---

### 4.4 Multi Stage Retrieval

### 4.4.1 Two-Stage Retrieval

Typical pipeline:

1. Fast retriever retrieves top K candidates
2. Reranker refines the list

First stage:

- BM25 or dense ANN search

Second stage:

- Cross-encoder
- LLM-based reranker

**Reasoning**

- First stage optimizes recall
- Second stage optimizes precision

---

### 4.4.2 Cross-Encoder Reranking

Cross-encoders jointly encode query and document.

**Advantages**

- Rich token-level interactions
- Higher ranking accuracy

**Limitations**

- Computationally expensive
- Cannot be applied to large corpora directly

---

### 4.5 Query-aware Retrieval Techniques

### 4.5.1 Query Expansion

Techniques:

- Synonym expansion
- LLM-generated query reformulations
- Multi-query retrieval

**Why it helps**

- Improves recall
- Handles ambiguous or underspecified queries

**Risk**

- Query drift
- Increased noise

---

### 4.5.2 Hypothetical Document Embeddings (HyDE)

HyDE generates a hypothetical answer and embeds it for retrieval.

**Reasoning**

- The generated answer is closer to relevant documents than the original query
- Improves semantic alignment

**Failure mode**

- Model hallucination propagates into retrieval

---

### 5. Re Ranking

### 5.1 Cross Encoder Re Ranking

#### Pros

- Very high precision
- Strong relevance modeling

### Cons

- Computationally expensive
- Limited to small candidate sets

---

### 5.2 LLM Based Re Ranking

#### Pros

- Flexible reasoning
- Can incorporate task specific instructions

#### Cons

- Expensive
- Non deterministic
- Prompt sensitive

---

### 6. Retrieval Evaluation Metrics

Common metrics:

- Recall@K
- Precision@K
- MRR
- nDCG

**Key reasoning**

High Recall@K is often more important than precision because the LLM can filter irrelevant context better than it can invent missing information.

---

### 7. System Design Tradeoffs

- **Latency vs Recall**: Deeper search improves recall but increases response time.
- **Memory vs Accuracy**: Compression saves memory at the cost of precision.
- **Static vs Dynamic Data**: Frequent updates favor graph based indexes.

---

### 8. Common Failure Modes

- Poor chunking leading to partial context
- Embedding mismatch between query and corpus
- Over aggressive filtering
- High recall but low precision hurting generation
- Untuned index parameters

---