### 1. Overview

Chunking is the process of splitting documents into smaller units before embedding and indexing. It is a critical design choice in RAG systems because it directly impacts retrieval quality, context relevance, latency, and cost.

Long documents must be broken down into chunks to:

- Fit model token limits
- Improve retrieval granularity
- Reduce semantic dilution
- Control prompt length and cost

Poor chunking can cause:

- Retrieval failure
- Loss of context
- Fragmented or incomplete answers

### 2. Chunking Strategies

### 2.1 Fixed-Size Chunking

Documents are split into chunks of a fixed token or character length, often with optional overlap.

Example:

- Chunk size: 512 tokens
- Overlap: 50 tokens

#### Algorithm

1. Tokenize the document
2. Split tokens into consecutive windows of size `N`
3. Optionally overlap adjacent windows by `M` tokens

#### Pros

- Simple to implement
- Fast and scalable
- Works reasonably well as a baseline

#### Cons

- Ignores semantic boundaries
- May split sentences or paragraphs
- Important information can be fragmented across chunks

#### When to Use

- Baseline RAG systems
- Uniform document formats
- Large-scale indexing where simplicity matters

---

### 2.2 Sentence-Based Chunking

Documents are split at sentence boundaries, often grouping multiple sentences into a single chunk until a token limit is reached.

#### Algorithm

1. Sentence tokenize the document
2. Accumulate sentences until the chunk reaches a size threshold
3. Start a new chunk when the threshold is exceeded

#### Pros

- Preserves sentence semantics
- Reduces mid-sentence splits
- Better coherence than fixed-size chunking

#### Cons

- Sentence lengths vary significantly
- Long sentences can exceed limits
- Still ignores higher-level structure

#### When to Use

- Narrative text
- QA over articles or reports
- Medium-length documents

---

### 2.3 Paragraph-Based Chunking

Chunks are formed using paragraph boundaries, typically defined by newline separation.

#### Algorithm

1. Split document by paragraph delimiters
2. Merge small paragraphs until size threshold is met
3. Split large paragraphs if needed

#### Pros

- Preserves local topical coherence
- Aligns well with human-written structure
- Better retrieval relevance for explanatory text

#### Cons

- Paragraph length is highly inconsistent
- Large paragraphs may require further splitting
- Formatting noise can affect quality

#### When to Use

- Well-structured documents
- Technical blogs and documentation
- Markdown or HTML content

---

### 2.4 Recursive Chunking

Recursive chunking applies a hierarchy of splitting rules, starting with coarse semantic units and progressively falling back to finer ones if size constraints are violated.

Typical split order:

- Sections
- Subsections
- Paragraphs
- Sentences
- Fixed-size fallback

#### Algorithm

1. Attempt to split by highest-level boundary
2. If chunk exceeds size limit, split using next level
3. Continue recursively until constraints are satisfied

#### Pros

- Preserves document structure
- Produces semantically meaningful chunks
- Handles diverse document formats well

#### Cons

- More complex to implement
- Requires reliable document parsing
- Slightly higher preprocessing cost

#### When to Use

- Enterprise documents
- PDFs with headings
- Mixed-format content

---

### 2.5 Context-Aware Chunking

Chunks are formed based on semantic similarity rather than fixed boundaries. Adjacent text segments are grouped if they share high semantic coherence.

#### Algorithm

1. Compute embeddings for small text units
2. Measure semantic similarity between adjacent units
3. Merge units until similarity drops or size limit is reached

#### Pros

- High semantic coherence
- Reduces context fragmentation
- Improves retrieval precision

#### Cons

- Computationally expensive
- Requires embedding during preprocessing
- Sensitive to similarity thresholds

#### When to Use

- High-accuracy RAG systems
- Knowledge-intensive QA
- Smaller corpora where quality matters

---

### 2.6 Sliding Window Chunking

Chunks are generated using overlapping windows that slide across the document.

Example:

- Window size: 512 tokens
- Stride: 256 tokens

#### Pros

- Preserves cross-boundary context
- Reduces information loss at chunk edges
- Improves recall

#### Cons

- Increased index size
- Higher storage and retrieval cost
- More redundant embeddings

#### When to Use

- Long-form documents
- Cases where boundary loss is critical
- Multi-hop reasoning tasks

---

## Chunking Strategy Comparison

| Strategy | Semantic Coherence | Complexity | Index Size | Common Use |
|-------|-------------------|------------|-----------|-----------|
| Fixed-size | Low | Low | Medium | Baselines |
| Sentence-based | Medium | Low | Medium | Articles |
| Paragraph-based | Medium | Low | Medium | Documentation |
| Recursive | High | Medium | Medium | Enterprise RAG |
| Context-aware | High | High | Low to Medium | High-precision RAG |
| Sliding window | Medium | Low | High | Long documents |

---


### 3. Other Concepts

#### 3.1 Chunk Size vs Top-k Tradeoffs

Chunk size and top-k retrieval are tightly coupled design parameters in RAG systems. Changing one almost always requires adjusting the other.

#### Key Intuition

- Smaller chunks increase retrieval granularity but reduce context per chunk
- Larger chunks provide more local context but reduce retrieval precision

#### Tradeoff Matrix

| Chunk Size | Typical Top-k | Behavior |
|-----------|--------------|----------|
| Small (100–300 tokens) | High (10–20) | High recall, lower precision |
| Medium (300–700 tokens) | Medium (4–8) | Balanced retrieval |
| Large (700–1500 tokens) | Low (1–3) | High precision, risk of misses |

#### Failure Patterns

- Small chunks + low top-k → missing required information
- Large chunks + high top-k → context overload
- Large chunks + low top-k → partial coverage

---

#### 3.2 Chunk Overlap Selection

Overlap determines how much content is shared between adjacent chunks.

#### Why Overlap Matters

- Prevents information loss at chunk boundaries
- Preserves cross-sentence and cross-paragraph context

#### Typical Settings

- Fixed-size chunking: 10 to 20 percent overlap
- Sliding window chunking: stride equals 50 percent of window size
- Recursive chunking: overlap often unnecessary

#### Tradeoffs

**Benefits**
- Improved recall
- Reduced boundary effects

**Costs**
- Larger index size
- Higher storage and retrieval cost
- More redundant embeddings

#### Insight

Overlap helps recall but should be used sparingly. Overlap is a mitigation strategy, not a substitute for good chunking.

---

#### 3.3 Chunk Metadata and Filtering

#### What Is Chunk Metadata?

Metadata is structured information attached to each chunk that enables filtering and ranking during retrieval.

**Common metadata fields**

- Document ID
- Section or heading
- Timestamp or version
- Author or source
- Content type

#### How Metadata Is Used

- Filter chunks before similarity search
- Re-rank retrieved chunks
- Restrict retrieval to specific document subsets

#### Benefits

- Improves retrieval precision
- Reduces noise
- Enables structured queries

#### Example

Retrieve only:

- Chunks from a specific product version
- Chunks created after a certain date
- Chunks from a trusted source

#### Insight

Metadata filtering is one of the most effective ways to improve vanilla RAG without changing models.

---

#### 3.4 Chunking for Tables and Code

Text-centric chunking often fails for structured content such as tables and source code.

#### 3.4.1 Chunking Tables

**Challenges**

- Rows and columns encode relationships
- Linear text chunking destroys structure

**Common strategies**

- Row-based chunking
- Column-wise chunking for analytical queries
- Table-to-text serialization with schema preservation

**Best practice**

Attach table schema and headers as metadata to every chunk.

#### 3.4.2 Chunking Code

**Challenges**

- Long-range dependencies
- Functions and classes are semantic units

**Common strategies**

- Function-level chunking
- Class-level chunking
- File-level chunking for small files

**Best practice**

Never split a function or class across chunks.

---

#### 3.5 Adaptive Chunking Based on Query Type

#### Motivation

Different queries require different chunking granularity. Static chunking cannot optimally serve all query types.

#### Common Query Types

| Query Type | Preferred Chunking |
|----------|-------------------|
| Fact lookup | Small chunks |
| Concept explanation | Medium chunks |
| Procedural steps | Large chunks |
| Multi-hop reasoning | Overlapping or sliding window chunks |

#### Adaptive Strategies

- Maintain multiple indexes with different chunk sizes
- Dynamically select top-k based on query intent
- Use query classification to select chunking policy

#### Pros

- Higher retrieval accuracy
- Better context utilization
- Reduced hallucinations

#### Cons

- Increased system complexity
- Higher indexing and storage cost
- Requires query understanding

---

### 4. Takeaways

- How do you choose the right chunk size?
   - The Embedding Model: Some models perform better with short sentences; others can handle long paragraphs.
   - User Query Style: If users ask short, specific questions, small chunks are better. If they ask for summaries, larger chunks are needed.
   - The "Lost in the Middle" Phenomenon: LLMs struggle to find information buried in the middle of a massive chunk.
- Chunking directly impacts retrieval quality and hallucination rates
- Smaller chunks improve recall, larger chunks improve coherence
- There is no universally optimal chunking strategy
- Recursive and context-aware chunking are commonly used in production
- Chunking should be evaluated jointly with retrieval and prompting
- Chunk size, overlap, and top-k must be tuned jointly
- Metadata filtering often yields large quality gains
- Tables and code require specialized chunking
- Adaptive chunking improves robustness but adds complexity

---