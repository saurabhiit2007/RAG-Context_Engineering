## Retrival Augmented Generation (RAG)

### 1. Overview

#### What Problem Does RAG Solve?

Large Language Models are trained on a fixed snapshot of data and store knowledge implicitly in their parameters. This leads to several limitations:

#### Key Limitations of Vanilla LLMs

1. **Knowledge cutoff**  
   Models cannot natively access information created after training.

2. **Hallucinations**  
   When the model lacks factual knowledge, it may generate fluent but incorrect answers.

3. **Poor domain specificity**  
   General models struggle with enterprise, proprietary, or niche domain data.

4. **Cost of knowledge updates**  
   Updating knowledge via retraining or fine-tuning is expensive and slow.

#### How RAG Addresses These Issues

RAG decouples **knowledge storage** from **language generation**. Instead of forcing the model to memorize facts, it retrieves relevant external information at inference time and conditions generation on that retrieved context.

This allows:
- Up-to-date knowledge access
- Reduced hallucinations
- Grounded and explainable outputs
- Faster and cheaper iteration cycles

---

### 2. RAG vs Fine-Tuning: When Does RAG Help?

#### When RAG Is the Better Choice

RAG is preferred when:

- Knowledge changes frequently such as policies, documentation, or news.
- Data is large, unstructured, or proprietary.
- Source grounding and explainability are important.
- You want to avoid frequent retraining costs.

**Common use cases**
- Enterprise knowledge assistants
- Question answering over PDFs, wikis, and tickets
- Customer support bots grounded in internal documents

#### When Fine-Tuning Is the Better Choice

Fine-tuning is preferred when:

- You want to change **model behavior**, not just factual knowledge.
- The task requires consistent style, tone, or reasoning patterns.
- The knowledge is stable and relatively small.

**Common use cases**
- Instruction following improvements
- Code style adaptation
- Domain-specific reasoning patterns

> In practice, RAG and fine-tuning are often combined. A model may be fine-tuned for instruction following and reasoning, while RAG provides factual grounding.

---

### 3. High-Level RAG Pipeline

A standard RAG system consists of four main stages:

1. Indexing  
2. Retrieval  
3. Augmentation  
4. Generation  

---

### 3.1 Indexing

Indexing prepares external knowledge for efficient retrieval at inference time.

**Steps in Indexing**

1. **Document Ingestion**:  Load raw documents such as PDFs, HTML, markdown files, logs, or database records.

2. **Chunking**: Split documents into smaller passages, typically 200 to 1000 tokens.
Chunking is necessary because:
   - Embedding models have fixed input limits
   - Smaller chunks improve retrieval granularity

3. **Embedding**: Each chunk is converted into a dense vector using an embedding model.
For each document chunk $d_i$:
$$
\mathbf{e}_i = f_{\text{embed}}(d_i)
$$

4. **Storage**: All embeddings are stored in a vector database or approximate nearest neighbor index.

---

### 3.2 Retrieval

Retrieval happens at query time and selects the most relevant chunks.

### Retrieval Process

1. Embed the user query $q$ into a vector $\mathbf{e}_q$
2. Compute similarity between the query vector and stored document vectors
3. Select the top-k most similar chunks

A common similarity metric is cosine similarity:
$$
\text{score}(q, d_i) =
\frac{\mathbf{e}_q \cdot \mathbf{e}_i}{\|\mathbf{e}_q\|\|\mathbf{e}_i\|}
$$

> Retrieval quality is often the single most important factor in RAG system performance.

---

### 3.3 Augmentation

Augmentation injects retrieved content into the model input.

### Common Augmentation Strategies

- Concatenating retrieved chunks before the question
- Adding separators, titles, or metadata
- Structuring context using bullet points or citations

---

### 3.4 Generation

The LLM generates the final response conditioned on both:

- The user query
- The retrieved external context

Formally:
$$
P(y \mid x) = P(y \mid q, d_{1:k})
$$

The model reasons over provided evidence rather than relying solely on its internal parameters.

---

### 4. Failure Modes of Vanilla RAG

#### 4.1 Retrieval Failures

#### Poor Recall
Relevant documents exist in the corpus but are not retrieved.

**Common causes**
- Weak or misaligned embedding model
- Poor chunking strategy
- Query embedding mismatch with document embeddings

**Impact**
- The model generates answers without the required evidence
- Hallucinations reappear despite using RAG

---

#### Poor Precision
Retrieved chunks are irrelevant or only loosely related to the query.

**Common causes**
- Overly large chunks
- Ambiguous user queries
- High top-k retrieval without filtering

**Impact**
- Model is distracted by irrelevant context
- Answers become vague or incorrect

---

#### 4.2 Chunking-Related Failures

#### Over-Chunking
Chunks are too small and lose semantic meaning.

**Impact**
- Missing context
- Incomplete or misleading information

#### Under-Chunking
Chunks are too large and contain multiple unrelated topics.

**Impact**
- Lower retrieval accuracy
- Context window pollution

> Chunk size is a critical hyperparameter and strongly influences retrieval quality.

---

#### 4.3 Context Window Limitations

Vanilla RAG typically concatenates retrieved chunks directly into the prompt.

**Failure cases**
- Retrieved context exceeds the model context window
- Important chunks are truncated
- Ordering of chunks affects answer quality

**Impact**
- Loss of critical evidence
- Inconsistent or incomplete answers

---

#### 4.4 Lack of Reasoning Over Retrieved Evidence

Vanilla RAG assumes the model will correctly use the retrieved context.

**Common issues**
- Model ignores relevant chunks
- Model cherry-picks incorrect information
- Model blends retrieved facts with hallucinated content

**Impact**
- Answers appear grounded but contain subtle errors

---

#### 4.5 Retrieval-Augmentation Mismatch

The retrieved information may be correct but poorly integrated into the prompt.

**Common causes**
- No clear separation between context and question
- Missing instructions to ground answers in retrieved content
- No citation or reference constraints

**Impact**
- Model treats retrieved text as optional background
- Reduced factual consistency

---

#### 4.6 Static Retrieval Strategy

Vanilla RAG uses a single retrieval step.

**Limitations**
- Cannot refine or reformulate queries
- No iterative retrieval based on intermediate reasoning
- No handling of multi-hop or compositional questions

**Impact**
- Poor performance on complex queries
- Failure on questions requiring reasoning across documents

---

#### 4.7 No Verification or Feedback Loop

Vanilla RAG pipelines usually lack validation of retrieved or generated content.

**Missing components**
- Answer verification
- Retrieval confidence estimation
- Self-correction mechanisms

**Impact**
- Silent failures
- Difficult to debug or evaluate system behavior

---

### 5. Key Takeaways

- RAG reduces hallucinations but does not eliminate them
- Retrieval quality is the dominant failure point
- Chunking and prompt structure are first-order design choices
- Vanilla RAG struggles with complex, multi-hop reasoning
- Advanced RAG systems add reranking, query rewriting, verification, and feedback loops

> Most real-world RAG systems extend vanilla RAG to explicitly address these failure modes.

---
