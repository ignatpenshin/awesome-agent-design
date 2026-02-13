# RAG Systems & Vector Databases

> *"The best model in the world is useless if it doesn't have access to the right information at the right time."*

---

Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding LLM outputs in factual, up-to-date information. Rather than fine-tuning a model on proprietary data — an expensive and quickly-stale approach — RAG systems retrieve relevant documents at query time and inject them into the model's context window.

But "retrieve and stuff into prompt" is only the beginning. Production RAG systems involve sophisticated chunking strategies, hybrid retrieval methods, re-ranking pipelines, query transformations, and purpose-built vector databases. This chapter covers the full spectrum: from naive implementations to modular, self-correcting architectures that operate reliably at scale.

## Three Generations of RAG

The evolution of RAG can be understood through three distinct generations, each addressing the limitations of its predecessor.

### Naive RAG

The first generation follows a straightforward pipeline:

```
Query → Retrieve → Generate
```

The user's query is embedded, a similarity search retrieves the top-k most similar document chunks, and those chunks are concatenated into the LLM's prompt alongside the original question.

```python
from openai import OpenAI
from qdrant_client import QdrantClient

client = OpenAI()
qdrant = QdrantClient("localhost", port=6333)

def naive_rag(query: str, collection: str = "documents") -> str:
    """Simplest possible RAG: embed → search → generate."""
    # Step 1: Embed the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Step 2: Retrieve top-k similar chunks
    results = qdrant.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=5
    )

    # Step 3: Build context and generate
    context = "\n\n".join([r.payload["text"] for r in results])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content
```

Naive RAG is easy to implement and works surprisingly well for simple use cases. However, it suffers from several critical limitations:

- **Low retrieval accuracy.** Embedding similarity alone often misses semantically relevant documents, especially when the query uses different vocabulary than the source material.
- **No reranking.** The top-k results from vector search are used as-is, without a second-pass relevance assessment.
- **Poor chunking.** Fixed-size text splits frequently break sentences mid-thought, losing context and confusing the retriever.
- **No query understanding.** The raw user query is embedded directly, even when it is ambiguous, multi-part, or poorly phrased.
- **Hallucination risk.** Without verification, the model may generate plausible-sounding answers that are not grounded in the retrieved context.

### Advanced RAG

The second generation introduces pre-retrieval and post-retrieval processing stages that dramatically improve quality:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ADVANCED RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌───────────────┐    ┌────────────────────┐   │
│  │  User     │───▶│  Query        │───▶│  Hybrid Retrieval  │   │
│  │  Query    │    │  Transform    │    │                    │   │
│  └──────────┘    │               │    │  ┌──────────────┐  │   │
│                  │ • Expansion   │    │  │ Dense Search  │  │   │
│                  │ • HyDE        │    │  │ (Embeddings)  │  │   │
│                  │ • Decompose   │    │  └──────┬───────┘  │   │
│                  │ • Step-back   │    │         │          │   │
│                  └───────────────┘    │  ┌──────▼───────┐  │   │
│                                      │  │   Fusion      │  │   │
│                                      │  │   (RRF)       │  │   │
│                                      │  └──────┬───────┘  │   │
│                                      │  ┌──────▼───────┐  │   │
│                                      │  │ Sparse Search │  │   │
│                                      │  │ (BM25)        │  │   │
│                                      │  └──────────────┘  │   │
│                                      └─────────┬──────────┘   │
│                                                │              │
│                                      ┌─────────▼──────────┐   │
│                                      │    Re-ranking       │   │
│                                      │  (Cross-Encoder)    │   │
│                                      └─────────┬──────────┘   │
│                                                │              │
│                                      ┌─────────▼──────────┐   │
│                                      │    Generation       │   │
│                                      │  (LLM + Context)    │   │
│                                      └────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Key improvements over Naive RAG:

- **Query transformation.** Before retrieval, the query is rewritten, expanded, or decomposed to improve recall.
- **Hybrid retrieval.** Combines dense (embedding) and sparse (BM25) search with Reciprocal Rank Fusion to capture both semantic and lexical matches.
- **Re-ranking.** A cross-encoder model re-scores the retrieved candidates, promoting the most relevant documents to the top.
- **Structured prompting.** The generation step uses carefully designed prompts with source attribution and instructions to stay grounded in the context.

### Modular RAG

The third generation treats the RAG pipeline as a composable system of interchangeable modules with feedback loops and adaptive routing:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODULAR RAG ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Route    │───▶│ Retrieve │───▶│  Grade   │───▶│ Generate │ │
│  └────┬─────┘    └──────────┘    └────┬─────┘    └────┬─────┘ │
│       │                               │               │       │
│       │ ┌─────────────────────────────┘               │       │
│       │ │  Irrelevant?                                │       │
│       │ ▼                                             │       │
│  ┌────┴─────┐                              ┌─────────▼─────┐ │
│  │ Web      │                              │  Hallucination │ │
│  │ Search   │                              │  Check         │ │
│  └──────────┘                              └─────────┬─────┘ │
│                                                      │       │
│                                            ┌─────────▼─────┐ │
│                                            │  Regenerate   │ │
│                                            │  if needed    │ │
│                                            └───────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Three influential modular RAG architectures have emerged:

**Self-RAG** introduces a self-reflection mechanism. The model decides whether retrieval is needed, evaluates whether retrieved documents are relevant, and checks whether its own generation is grounded in the evidence. Special "reflection tokens" guide these decisions.

**Corrective RAG (CRAG)** adds a retrieval evaluator that grades each document's relevance. If documents are ambiguous or irrelevant, the system triggers a web search to supplement or replace the retrieved context before generation.

**Adaptive RAG** dynamically selects the retrieval strategy based on query complexity. Simple factual queries may bypass retrieval entirely (relying on the model's parametric knowledge), moderate queries use single-step RAG, and complex queries trigger multi-step retrieval with iterative refinement.

```python
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class QueryComplexity(str, Enum):
    SIMPLE = "simple"        # Answer from parametric knowledge
    MODERATE = "moderate"    # Single-step RAG
    COMPLEX = "complex"      # Multi-step retrieval

class QueryAnalysis(BaseModel):
    complexity: QueryComplexity
    reasoning: str
    sub_queries: list[str] = []

def adaptive_rag(query: str) -> str:
    """Route query to appropriate RAG strategy based on complexity."""
    # Step 1: Analyze query complexity
    analysis_response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Analyze the query complexity:\n"
                "- SIMPLE: Factual, well-known, single-fact answer\n"
                "- MODERATE: Requires context but single retrieval step\n"
                "- COMPLEX: Multi-faceted, needs multiple retrievals"
            )},
            {"role": "user", "content": query}
        ],
        response_format=QueryAnalysis
    )
    analysis = analysis_response.choices[0].message.parsed

    # Step 2: Route to appropriate strategy
    match analysis.complexity:
        case QueryComplexity.SIMPLE:
            return generate_without_retrieval(query)
        case QueryComplexity.MODERATE:
            return single_step_rag(query)
        case QueryComplexity.COMPLEX:
            return multi_step_rag(query, analysis.sub_queries)
```

The modular approach allows teams to swap components independently — replacing one embedding model with another, adding a new reranker, or introducing a verification step — without rebuilding the entire pipeline.

---

## Chunking Strategies

Chunking is the process of splitting documents into segments suitable for embedding and retrieval. The choice of chunking strategy has a profound impact on retrieval quality: chunks that are too small lose context, while chunks that are too large dilute relevance and waste precious context window tokens.

### 1. Fixed-Size Chunking

The simplest approach: split text into chunks of a fixed number of characters (or tokens) with optional overlap.

```python
def fixed_size_chunking(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[str]:
    """Split text into fixed-size chunks with overlap.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Example
text = "A" * 2500  # 2500 characters
chunks = fixed_size_chunking(text, chunk_size=1000, overlap=200)
# Result: 4 chunks, each ≤1000 chars, 200-char overlap between neighbors
```

**Pros:** Dead simple, predictable chunk sizes, fast.
**Cons:** Splits words and sentences mid-thought, ignores document structure, overlap wastes storage.

### 2. Recursive Chunking

LangChain's `RecursiveCharacterTextSplitter` attempts to split text along natural boundaries — paragraphs first, then sentences, then words — falling back to finer-grained separators only when a chunk exceeds the target size.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",   # First try: split on paragraphs
        "\n",     # Then: line breaks
        ". ",     # Then: sentences
        ", ",     # Then: clauses
        " ",      # Then: words
        ""        # Last resort: characters
    ],
    length_function=len
)

text = """
Introduction to Machine Learning

Machine learning is a branch of artificial intelligence
focused on building systems that learn from data.

Supervised Learning

In supervised learning, the model is trained on labeled data.
The algorithm learns to map inputs to outputs based on
example input-output pairs.

Unsupervised Learning

In unsupervised learning, the model works with unlabeled data.
It must discover structure and patterns on its own.
"""

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:80]}...")
```

The recursive strategy respects document structure far better than fixed-size chunking. It keeps paragraphs intact when possible and only breaks them when they exceed the target size.

### 3. Semantic Chunking

Semantic chunking uses embedding similarity to determine where to split. The idea is that sentences about the same topic will have similar embeddings; a sharp drop in similarity between consecutive sentences signals a topic boundary.

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunking(
    text: str,
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 100
) -> list[str]:
    """Split text into chunks based on semantic similarity between sentences.

    Groups consecutive sentences that are semantically similar. When similarity
    drops below the threshold, a new chunk begins.

    Args:
        text: The input text to chunk.
        similarity_threshold: Minimum cosine similarity to keep sentences
            in the same chunk.
        min_chunk_size: Minimum character count before allowing a split.

    Returns:
        A list of semantically coherent text chunks.
    """
    # Split into sentences
    sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # Get embeddings for all sentences
    embeddings = get_embeddings(sentences)

    # Group sentences by semantic similarity
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

        current_text = ". ".join(current_chunk)
        if similarity < similarity_threshold and len(current_text) >= min_chunk_size:
            # Similarity dropped — start a new chunk
            chunks.append(current_text + ".")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks
```

Semantic chunking produces the most coherent chunks because it splits on actual topic boundaries rather than arbitrary character counts. The trade-off is cost: every sentence requires an embedding API call.

### 4. Document-Aware Chunking

For structured documents (Markdown, HTML, code), the most effective strategy is to respect the document's own structure — splitting on headers, sections, and logical boundaries.

```python
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """A chunk with its structural metadata."""
    content: str
    metadata: dict

def markdown_chunking(
    markdown_text: str,
    max_chunk_size: int = 1500
) -> list[DocumentChunk]:
    """Split Markdown text along header boundaries.

    Preserves the document hierarchy by tracking which section
    each chunk belongs to.

    Args:
        markdown_text: The Markdown-formatted input text.
        max_chunk_size: Maximum character count per chunk.

    Returns:
        A list of DocumentChunk objects with structural metadata.
    """
    chunks = []
    current_headers: dict[int, str] = {}
    current_content: list[str] = []

    for line in markdown_text.split("\n"):
        # Detect Markdown headers
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            # Save previous section as a chunk
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    chunks.append(DocumentChunk(
                        content=content,
                        metadata={
                            "headers": dict(current_headers),
                            "level": max(current_headers.keys())
                                   if current_headers else 0
                        }
                    ))
                current_content = []

            # Update header hierarchy
            level = len(header_match.group(1))
            title = header_match.group(2)
            current_headers[level] = title
            # Remove deeper-level headers (entering a new section)
            for deeper_level in list(current_headers.keys()):
                if deeper_level > level:
                    del current_headers[deeper_level]

        current_content.append(line)

    # Final section
    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            chunks.append(DocumentChunk(
                content=content,
                metadata={
                    "headers": dict(current_headers),
                    "level": max(current_headers.keys())
                               if current_headers else 0
                }
            ))

    # Split oversized chunks further
    final_chunks = []
    for chunk in chunks:
        if len(chunk.content) > max_chunk_size:
            sub_parts = split_preserving_sentences(
                chunk.content, max_chunk_size
            )
            for part in sub_parts:
                final_chunks.append(DocumentChunk(
                    content=part,
                    metadata=chunk.metadata
                ))
        else:
            final_chunks.append(chunk)

    return final_chunks

def split_preserving_sentences(text: str, max_size: int) -> list[str]:
    """Split text on sentence boundaries without exceeding max_size."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_size and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        parts.append(current.strip())
    return parts
```

### Chunking Strategy Comparison

| Strategy | Best For | Chunk Quality | Speed | Cost |
|----------|----------|---------------|-------|------|
| **Fixed-size** | Prototyping, uniform text | Low | Very fast | Minimal |
| **Recursive** | General-purpose documents | Medium-High | Fast | Minimal |
| **Semantic** | Mixed-topic documents | Highest | Slow | High (embeddings) |
| **Document-aware** | Structured docs (MD, HTML, code) | High | Fast | Minimal |

**Production recommendation:** Start with recursive chunking at 500-1000 tokens with 10-20% overlap. Move to semantic or document-aware chunking when retrieval quality needs improvement. For structured documents, always prefer document-aware chunking.

---

## Retrieval Methods

Retrieval is the core of any RAG system. The choice of retrieval method determines what information the LLM has access to — and therefore the quality of its responses.

### Dense Retrieval

Dense retrieval uses neural embedding models to represent both queries and documents as high-dimensional vectors. Retrieval is performed via approximate nearest neighbor (ANN) search in the vector space.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from openai import OpenAI

client = OpenAI()
qdrant = QdrantClient("localhost", port=6333)

def create_collection(name: str, vector_size: int = 1536):
    """Create a Qdrant collection for dense retrieval."""
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

def embed_text(text: str) -> list[float]:
    """Get embedding vector for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def index_documents(collection: str, documents: list[dict]):
    """Index documents with their embeddings into Qdrant.

    Args:
        collection: Name of the Qdrant collection.
        documents: List of dicts with 'id', 'text', and optional metadata.
    """
    points = []
    for doc in documents:
        embedding = embed_text(doc["text"])
        points.append(PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={
                "text": doc["text"],
                "source": doc.get("source", ""),
                "category": doc.get("category", ""),
            }
        ))

    qdrant.upsert(collection_name=collection, points=points)

def dense_search(query: str, collection: str, top_k: int = 10) -> list[dict]:
    """Search for similar documents using dense embeddings.

    Returns:
        List of dicts with 'text', 'score', and metadata.
    """
    query_embedding = embed_text(query)

    results = qdrant.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=top_k
    )

    return [
        {
            "text": r.payload["text"],
            "score": r.score,
            "source": r.payload.get("source", ""),
        }
        for r in results
    ]
```

Dense retrieval excels at **semantic matching** — finding documents that are conceptually related to the query, even when they use completely different vocabulary. Its weakness is **exact keyword matching**: a dense model may fail to retrieve a document containing a specific product name, error code, or identifier if it was underrepresented in the embedding model's training data.

### Sparse Retrieval (BM25)

BM25 is a classical information retrieval algorithm that scores documents based on term frequency, inverse document frequency, and document length normalization. It excels at finding documents that share specific keywords with the query.

```python
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    """Sparse retriever using the BM25 algorithm.

    BM25 is a bag-of-words ranking function that scores documents
    based on term frequency and inverse document frequency. It is
    particularly strong for keyword-centric queries.
    """

    def __init__(self, documents: list[dict]):
        """Initialize with a corpus of documents.

        Args:
            documents: List of dicts, each with at least a 'text' key.
        """
        self.documents = documents
        # Tokenize documents (simple whitespace split; production
        # systems should use a proper tokenizer)
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the corpus for documents matching the query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'text', 'score', and original metadata.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top-k scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.documents[i],
                "score": float(scores[i])
            }
            for i in top_indices
            if scores[i] > 0
        ]
```

BM25 is complementary to dense retrieval: it catches the exact-match cases that embeddings miss, while embeddings catch the semantic cases that BM25 misses.

### Hybrid Search

The best production RAG systems combine dense and sparse retrieval using **Reciprocal Rank Fusion (RRF)**, a simple but effective algorithm that merges ranked lists from multiple sources.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF assigns a score of 1/(k + rank) to each item from each list,
    then sums scores across lists. The constant k (default 60) controls
    how much weight is given to lower-ranked items.

    Args:
        ranked_lists: List of ranked result lists. Each result dict
            must contain a 'text' key for deduplication.
        k: RRF constant. Higher values reduce the influence of rank.

    Returns:
        A single merged and re-ranked list of results.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_key = doc["text"][:200]  # Use text prefix as dedup key
            fused_scores[doc_key] = fused_scores.get(doc_key, 0) + 1 / (k + rank + 1)
            doc_map[doc_key] = doc

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)

    return [
        {**doc_map[key], "rrf_score": fused_scores[key]}
        for key in sorted_keys
    ]

def hybrid_search(
    query: str,
    collection: str,
    bm25_retriever: BM25Retriever,
    top_k: int = 10
) -> list[dict]:
    """Perform hybrid search combining dense and sparse retrieval.

    Runs both dense (embedding) and sparse (BM25) searches in parallel,
    then merges results using Reciprocal Rank Fusion.

    Args:
        query: The search query string.
        collection: Name of the Qdrant collection for dense search.
        bm25_retriever: An initialized BM25Retriever instance.
        top_k: Number of results to return after fusion.

    Returns:
        Fused and ranked list of results.
    """
    # Get results from both retrieval methods
    dense_results = dense_search(query, collection, top_k=20)
    sparse_results = bm25_retriever.search(query, top_k=20)

    # Merge with RRF
    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    return fused[:top_k]
```

Hybrid search consistently outperforms either method in isolation across benchmarks and real-world workloads. The dense retriever provides semantic understanding while BM25 provides precise keyword matching — together they cover the full spectrum of query types.

---

## Re-ranking

Retrieval returns a candidate set. Re-ranking reorders those candidates using a more powerful (and more expensive) model to push the most relevant documents to the top.

The most common approach uses a **cross-encoder**: a model that takes the query and a document as a single input and outputs a relevance score. Unlike bi-encoders (used for embedding), cross-encoders attend to the full interaction between query and document, making them far more accurate — but too slow to run over an entire corpus.

This is why re-ranking is a second-pass operation: the retriever narrows millions of documents to a few dozen candidates, and the re-ranker selects the best among them.

```python
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    """Cross-encoder reranker for improving retrieval precision.

    Uses a cross-encoder model to jointly score query-document pairs,
    capturing fine-grained relevance signals that bi-encoder embeddings miss.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """Initialize with a cross-encoder model.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
        """
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        """Rerank documents by relevance to the query.

        Args:
            query: The original user query.
            documents: List of candidate documents from the retriever.
            top_k: Number of top-scoring documents to return.

        Returns:
            Top-k documents re-ordered by cross-encoder relevance score.
        """
        if not documents:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

# Usage in a pipeline
reranker = Reranker()

def retrieval_pipeline(query: str, collection: str, bm25: BM25Retriever) -> list[dict]:
    """Full retrieval pipeline: hybrid search → rerank → top results."""
    # Step 1: Retrieve candidates via hybrid search
    candidates = hybrid_search(query, collection, bm25, top_k=20)

    # Step 2: Rerank to get the most relevant documents
    top_docs = reranker.rerank(query, candidates, top_k=5)

    return top_docs
```

**Production considerations for re-ranking:**

- **Retrieve broadly, rerank narrowly.** Retrieve 20-50 candidates, rerank to 3-5. This maximizes recall while keeping the final context focused.
- **Model selection.** `ms-marco-MiniLM-L-12-v2` is an excellent balance of speed and quality. For higher accuracy, consider `cross-encoder/ms-marco-MiniLM-L-6-v2` (faster) or Cohere Rerank API (higher quality, managed service).
- **Latency budget.** Cross-encoder inference on 20 documents typically takes 50-200ms on CPU. For real-time applications, consider GPU inference or a hosted reranking API.

---

## Query Transformation

Raw user queries are often suboptimal for retrieval. Query transformation techniques rewrite, expand, or decompose queries to improve recall and precision.

### HyDE (Hypothetical Document Embeddings)

HyDE asks the LLM to generate a hypothetical answer to the query, then uses the embedding of that hypothetical answer as the search vector. The intuition is that a hypothetical answer will be closer in embedding space to the actual answer document than the original question is.

```python
from openai import OpenAI

client = OpenAI()

def hyde_search(
    query: str,
    collection: str,
    top_k: int = 10
) -> list[dict]:
    """Search using Hypothetical Document Embeddings (HyDE).

    Instead of embedding the raw query, we first generate a hypothetical
    answer, then embed and search with that. The hypothetical answer
    uses vocabulary and phrasing closer to actual source documents.

    Args:
        query: The original user question.
        collection: Name of the Qdrant collection.
        top_k: Number of results to return.

    Returns:
        List of retrieved documents ranked by similarity to the
        hypothetical answer embedding.
    """
    # Step 1: Generate a hypothetical answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Write a detailed paragraph that would answer "
                "the following question. Write as if you are "
                "writing a passage from a technical document."
            )},
            {"role": "user", "content": query}
        ],
        max_tokens=300
    )
    hypothetical_answer = response.choices[0].message.content

    # Step 2: Embed the hypothetical answer (not the original query)
    hypothesis_embedding = embed_text(hypothetical_answer)

    # Step 3: Search with the hypothesis embedding
    results = qdrant.search(
        collection_name=collection,
        query_vector=hypothesis_embedding,
        limit=top_k
    )

    return [
        {
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]
```

HyDE is particularly effective for questions where the query and the answer use fundamentally different vocabulary — for example, "What causes servers to become unresponsive?" might retrieve documents about "resource exhaustion," "memory leaks," and "connection pool saturation" more effectively via a hypothetical answer than via the raw question.

### Query Decomposition

Complex multi-part questions often cannot be answered by a single retrieval step. Query decomposition breaks a complex question into simpler sub-queries, retrieves for each, and synthesizes the results.

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class DecomposedQuery(BaseModel):
    sub_queries: list[str]
    reasoning: str

def decompose_query(query: str) -> list[str]:
    """Decompose a complex query into simpler sub-queries.

    Uses an LLM to analyze the question and break it into independent
    sub-questions that can each be answered with a single retrieval step.

    Args:
        query: A complex, multi-faceted question.

    Returns:
        A list of simpler sub-queries.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Break down the complex question into 2-4 simpler, "
                "independent sub-questions. Each sub-question should be "
                "answerable with a single retrieval step."
            )},
            {"role": "user", "content": query}
        ],
        response_format=DecomposedQuery
    )
    return response.choices[0].message.parsed.sub_queries

def multi_step_rag(query: str, sub_queries: list[str]) -> str:
    """Answer a complex query by decomposing, retrieving, and synthesizing.

    Args:
        query: The original complex question.
        sub_queries: Pre-decomposed sub-queries.

    Returns:
        A synthesized answer drawing from all sub-query results.
    """
    all_contexts = []

    for sub_query in sub_queries:
        # Retrieve for each sub-query independently
        results = dense_search(sub_query, "documents", top_k=3)
        context = "\n".join([r["text"] for r in results])
        all_contexts.append(f"Sub-question: {sub_query}\nContext: {context}")

    combined_context = "\n\n---\n\n".join(all_contexts)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Synthesize a comprehensive answer from the "
                "provided sub-question contexts. Cite which "
                "sub-question each piece of information comes from."
            )},
            {"role": "user", "content": (
                f"Original question: {query}\n\n{combined_context}"
            )}
        ]
    )
    return response.choices[0].message.content

# Example
query = "Compare the performance of PostgreSQL and MongoDB for write-heavy workloads and how does each handle horizontal scaling?"
sub_queries = decompose_query(query)
# Possible decomposition:
# 1. "What is PostgreSQL's write performance and optimization strategies?"
# 2. "What is MongoDB's write performance and optimization strategies?"
# 3. "How does PostgreSQL handle horizontal scaling?"
# 4. "How does MongoDB handle horizontal scaling?"
```

### Step-back Prompting

Step-back prompting generates a more general, higher-level version of the query before retrieval. By "stepping back" from the specific question, the retriever can find broader context documents that contain the necessary background information.

```python
from openai import OpenAI

client = OpenAI()

def step_back_search(
    query: str,
    collection: str,
    top_k: int = 10
) -> list[dict]:
    """Retrieve using a step-back (more general) version of the query.

    Some specific questions are best answered by first retrieving
    broader contextual information. Step-back prompting generates a
    more abstract version of the query for retrieval.

    Args:
        query: The original specific question.
        collection: Name of the vector collection.
        top_k: Number of results to return.

    Returns:
        Combined results from both the original and step-back queries.
    """
    # Step 1: Generate a step-back question
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Given a specific question, generate a more general "
                "step-back question that would help retrieve broader "
                "contextual information needed to answer the original. "
                "Return only the step-back question."
            )},
            {"role": "user", "content": query}
        ],
        max_tokens=100
    )
    step_back_query = response.choices[0].message.content

    # Step 2: Retrieve for both the original and step-back queries
    original_results = dense_search(query, collection, top_k=top_k // 2)
    step_back_results = dense_search(step_back_query, collection, top_k=top_k // 2)

    # Step 3: Merge results (RRF or simple concatenation)
    combined = reciprocal_rank_fusion([original_results, step_back_results])
    return combined[:top_k]

# Example
# Original:  "Why does my CUDA kernel fail with error code 700?"
# Step-back: "What are common CUDA kernel execution errors and their causes?"
```

Step-back prompting is especially useful in technical domains where understanding the broader concept is necessary to answer a specific question.

---

## Evaluation — RAGAS Framework

Measuring RAG quality requires evaluating both the retrieval and generation components. The RAGAS (Retrieval-Augmented Generation Assessment) framework provides four complementary metrics:

- **Faithfulness:** Does the generated answer stick to the retrieved context? Measures how many claims in the answer can be traced back to the provided documents.
- **Answer Relevancy:** Does the answer actually address the question? Measures alignment between the question asked and the answer given.
- **Context Precision:** Are the retrieved documents relevant? Measures how many of the retrieved documents are actually useful for answering the question.
- **Context Recall:** Did the retriever find all necessary information? Measures whether the retrieved context covers all the facts needed for a complete answer.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

def evaluate_rag_pipeline(
    questions: list[str],
    ground_truths: list[str],
    answers: list[str],
    contexts: list[list[str]]
) -> dict:
    """Evaluate a RAG pipeline using the RAGAS framework.

    Args:
        questions: List of input questions.
        ground_truths: List of reference (gold-standard) answers.
        answers: List of answers generated by the RAG pipeline.
        contexts: List of retrieved context lists (one per question).

    Returns:
        Dictionary of metric names to scores.
    """
    # Build the evaluation dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    })

    # Run RAGAS evaluation
    results = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return results

# Example evaluation
questions = [
    "What is the capital of France?",
    "How does photosynthesis work?",
]
ground_truths = [
    "The capital of France is Paris.",
    "Photosynthesis converts light energy into chemical energy using "
    "chlorophyll in plant cells.",
]
answers = [
    "Paris is the capital of France.",
    "Photosynthesis is the process by which plants convert sunlight "
    "into energy using chlorophyll.",
]
contexts = [
    ["France is a country in Western Europe. Its capital is Paris, "
     "which is also the largest city in France."],
    ["Photosynthesis is a biological process used by plants. "
     "Chlorophyll absorbs light energy and converts CO2 and water "
     "into glucose and oxygen."],
]

results = evaluate_rag_pipeline(questions, ground_truths, answers, contexts)

print("RAGAS Evaluation Results:")
print(f"  Faithfulness:       {results['faithfulness']:.3f}")
print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
print(f"  Context Precision:  {results['context_precision']:.3f}")
print(f"  Context Recall:     {results['context_recall']:.3f}")

# Target scores for production systems:
#   Faithfulness:      > 0.85
#   Answer Relevancy:  > 0.80
#   Context Precision: > 0.75
#   Context Recall:    > 0.80
```

**Beyond RAGAS:** While RAGAS provides a solid automated evaluation foundation, production systems should also incorporate:

- **A/B testing** to compare pipeline changes against a live baseline.
- **Human evaluation** for nuanced quality assessment, especially in high-stakes domains.
- **Domain-specific metrics** such as citation accuracy, regulatory compliance, or factual verification against authoritative sources.

---

## Vector Databases in Depth

Vector databases are purpose-built storage systems optimized for high-dimensional similarity search. While the RAG examples above used simple search calls, production deployments require careful attention to indexing, filtering, quantization, and hybrid search capabilities.

### Qdrant

Qdrant is a purpose-built vector database written in Rust, designed for high-throughput, low-latency similarity search at scale.

#### Collection Creation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SparseVector, SparseVectorParams, SparseIndexParams,
    ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, ProductQuantization, ProductQuantizationConfig,
    CompressionRatio, NamedSparseVector, NamedVector,
    SearchRequest,
)

client = QdrantClient("localhost", port=6333)

# Create a collection with vector configuration
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(
        size=1536,                    # Dimension of text-embedding-3-small
        distance=Distance.COSINE      # Similarity metric
    )
)
```

#### Data Insertion

```python
import uuid

def insert_documents(documents: list[dict]):
    """Insert documents with embeddings and metadata into Qdrant.

    Args:
        documents: List of dicts with 'text', 'embedding', and metadata fields.
    """
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=doc["embedding"],
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "category": doc["category"],
                "created_at": doc["created_at"],
                "chunk_index": doc.get("chunk_index", 0),
            }
        )
        for doc in documents
    ]

    # Upsert in batches for large datasets
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name="knowledge_base",
            points=points[i:i + batch_size]
        )
```

#### Search with Payload Filtering

Qdrant's payload filtering allows combining vector similarity with structured metadata conditions — essential for multi-tenant systems, date-range queries, and category-scoped searches.

```python
def filtered_search(
    query_embedding: list[float],
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 10
) -> list[dict]:
    """Search with combined vector similarity and payload filters.

    Args:
        query_embedding: The query vector.
        category: Optional category filter (exact match).
        date_from: Optional start date (ISO format, inclusive).
        date_to: Optional end date (ISO format, inclusive).
        top_k: Number of results to return.

    Returns:
        Filtered and ranked search results.
    """
    conditions = []

    if category:
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        )

    if date_from or date_to:
        range_params = {}
        if date_from:
            range_params["gte"] = date_from
        if date_to:
            range_params["lte"] = date_to

        conditions.append(
            FieldCondition(
                key="created_at",
                range=Range(**range_params)
            )
        )

    search_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=top_k
    )

    return [
        {
            "text": r.payload["text"],
            "score": r.score,
            "category": r.payload.get("category"),
            "source": r.payload.get("source"),
        }
        for r in results
    ]
```

#### Hybrid Search with Sparse Vectors

Qdrant natively supports sparse vectors, enabling built-in hybrid search without an external BM25 index.

```python
from qdrant_client.models import models

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="hybrid_knowledge_base",
    vectors_config={
        "dense": VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)

def insert_hybrid(documents: list[dict]):
    """Insert documents with both dense and sparse vector representations.

    Args:
        documents: List of dicts with 'text', 'dense_embedding',
            'sparse_indices', and 'sparse_values'.
    """
    points = [
        PointStruct(
            id=idx,
            vector={
                "dense": doc["dense_embedding"],
                "sparse": SparseVector(
                    indices=doc["sparse_indices"],
                    values=doc["sparse_values"]
                )
            },
            payload={"text": doc["text"]}
        )
        for idx, doc in enumerate(documents)
    ]
    client.upsert(collection_name="hybrid_knowledge_base", points=points)

def hybrid_search_qdrant(
    query_dense: list[float],
    query_sparse_indices: list[int],
    query_sparse_values: list[float],
    top_k: int = 10
) -> list[dict]:
    """Perform hybrid search using both dense and sparse vectors in Qdrant.

    Uses Qdrant's built-in query API to combine both signal types.

    Args:
        query_dense: Dense embedding vector for the query.
        query_sparse_indices: Sparse vector indices for the query.
        query_sparse_values: Sparse vector values for the query.
        top_k: Number of results to return.

    Returns:
        Results ranked by combined dense + sparse similarity.
    """
    # Search with dense vectors
    dense_results = client.search(
        collection_name="hybrid_knowledge_base",
        query_vector=NamedVector(name="dense", vector=query_dense),
        limit=top_k
    )

    # Search with sparse vectors
    sparse_results = client.search(
        collection_name="hybrid_knowledge_base",
        query_vector=NamedSparseVector(
            name="sparse",
            vector=SparseVector(
                indices=query_sparse_indices,
                values=query_sparse_values
            )
        ),
        limit=top_k
    )

    # Fuse results with RRF
    dense_list = [{"text": r.payload["text"], "score": r.score} for r in dense_results]
    sparse_list = [{"text": r.payload["text"], "score": r.score} for r in sparse_results]

    return reciprocal_rank_fusion([dense_list, sparse_list])[:top_k]
```

#### Quantization

Quantization reduces memory usage and increases search throughput by compressing vectors. Qdrant supports two quantization methods:

```python
# Scalar Quantization (INT8)
# Converts 32-bit floats to 8-bit integers.
# ~4x memory reduction with minimal accuracy loss.
client.create_collection(
    collection_name="quantized_scalar",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,         # Clip outliers beyond 99th percentile
            always_ram=True        # Keep quantized vectors in RAM
        )
    )
)

# Product Quantization
# Divides vectors into subvectors and quantizes each independently.
# Higher compression ratio but more accuracy loss.
client.create_collection(
    collection_name="quantized_product",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=CompressionRatio.X16,  # 16x compression
            always_ram=True
        )
    )
)
```

**Quantization guidance:**

| Method | Memory Savings | Accuracy Loss | Best For |
|--------|---------------|---------------|----------|
| Scalar (INT8) | ~4x | <1% | Default choice for production |
| Product (x16) | ~16x | 3-5% | Very large datasets (100M+ vectors) |
| Product (x32) | ~32x | 5-10% | Cost-constrained, recall-tolerant |

For most production workloads, scalar quantization provides the best trade-off: nearly identical accuracy with a 4x reduction in memory footprint.

### pgvector (PostgreSQL)

pgvector extends PostgreSQL with vector similarity search, enabling teams to store embeddings alongside their relational data without operating a separate vector database.

#### Extension Setup and Table Creation

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for documents with vector embeddings
CREATE TABLE documents (
    id            BIGSERIAL PRIMARY KEY,
    content       TEXT NOT NULL,
    embedding     vector(1536),           -- Matches text-embedding-3-small
    source        VARCHAR(500),
    category      VARCHAR(100),
    created_at    TIMESTAMP DEFAULT NOW(),

    -- Full-text search support
    content_tsv   tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', content)
                  ) STORED
);

-- Create full-text search index
CREATE INDEX idx_documents_tsv ON documents USING GIN(content_tsv);
```

#### Index Types

pgvector supports two index types for approximate nearest neighbor search:

```sql
-- HNSW Index (Recommended)
-- Hierarchical Navigable Small World graph.
-- Higher build time and memory, but faster and more accurate searches.
CREATE INDEX idx_documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- IVFFlat Index
-- Inverted file index with flat (exact) search within clusters.
-- Faster to build, lower memory, but less accurate than HNSW.
CREATE INDEX idx_documents_embedding_ivfflat
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);  -- Number of clusters; rule of thumb: sqrt(num_rows)
```

**Index comparison:**

| Aspect | HNSW | IVFFlat |
|--------|------|---------|
| Search speed | Faster | Slower |
| Build time | Slower | Faster |
| Memory usage | Higher | Lower |
| Recall accuracy | Higher (~99%) | Lower (~95%) |
| Requires training | No | Yes (cluster assignment) |
| **Recommendation** | **Default choice** | Budget-constrained, infrequent updates |

#### Vector Search with Cosine Distance

```sql
-- Basic vector similarity search
-- The <=> operator computes cosine distance (1 - cosine_similarity).
SELECT
    id,
    content,
    source,
    1 - (embedding <=> $1::vector) AS similarity
FROM documents
ORDER BY embedding <=> $1::vector
LIMIT 10;

-- Filtered vector search
SELECT
    id,
    content,
    source,
    1 - (embedding <=> $1::vector) AS similarity
FROM documents
WHERE category = 'technical'
  AND created_at >= '2024-01-01'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

#### Hybrid Search: Vector + Full-Text

One of pgvector's strongest advantages is seamless integration with PostgreSQL's built-in full-text search, enabling hybrid retrieval in a single SQL query:

```sql
-- Hybrid search combining vector similarity and full-text relevance
-- Uses RRF-style combination of both scores
WITH vector_search AS (
    SELECT
        id,
        content,
        source,
        ROW_NUMBER() OVER (
            ORDER BY embedding <=> $1::vector
        ) AS vector_rank
    FROM documents
    ORDER BY embedding <=> $1::vector
    LIMIT 20
),
text_search AS (
    SELECT
        id,
        content,
        source,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', $2)) DESC
        ) AS text_rank
    FROM documents
    WHERE content_tsv @@ websearch_to_tsquery('english', $2)
    LIMIT 20
)
SELECT
    COALESCE(v.id, t.id) AS id,
    COALESCE(v.content, t.content) AS content,
    COALESCE(v.source, t.source) AS source,
    -- Reciprocal Rank Fusion score
    COALESCE(1.0 / (60 + v.vector_rank), 0) +
    COALESCE(1.0 / (60 + t.text_rank), 0) AS rrf_score
FROM vector_search v
FULL OUTER JOIN text_search t ON v.id = t.id
ORDER BY rrf_score DESC
LIMIT 10;
```

#### Python asyncpg Example

```python
import asyncpg
import numpy as np
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()

class PgVectorStore:
    """Async pgvector-based document store using asyncpg.

    Provides methods for inserting, searching, and hybrid-searching
    documents stored in PostgreSQL with pgvector embeddings.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def create(cls, dsn: str) -> "PgVectorStore":
        """Factory method: create a PgVectorStore with a connection pool.

        Args:
            dsn: PostgreSQL connection string.

        Returns:
            An initialized PgVectorStore instance.
        """
        pool = await asyncpg.create_pool(dsn, min_size=5, max_size=20)

        # Register the vector type codec so asyncpg can handle vectors
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        return cls(pool)

    async def insert_document(
        self,
        content: str,
        embedding: list[float],
        source: str = "",
        category: str = ""
    ) -> int:
        """Insert a single document with its embedding.

        Args:
            content: The document text.
            embedding: The precomputed embedding vector.
            source: Optional source identifier.
            category: Optional category label.

        Returns:
            The ID of the inserted row.
        """
        async with self.pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO documents (content, embedding, source, category)
                VALUES ($1, $2::vector, $3, $4)
                RETURNING id
                """,
                content,
                str(embedding),  # pgvector accepts string representation
                source,
                category,
            )
        return row_id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        category: str | None = None
    ) -> list[dict]:
        """Vector similarity search with optional category filter.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            category: Optional category to filter by.

        Returns:
            List of matching documents with similarity scores.
        """
        async with self.pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    """
                    SELECT id, content, source, category,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM documents
                    WHERE category = $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding), top_k, category,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, content, source, category,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM documents
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    str(query_embedding), top_k,
                )

        return [dict(row) for row in rows]

    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10
    ) -> list[dict]:
        """Hybrid search combining vector similarity and full-text search.

        Uses Reciprocal Rank Fusion to merge vector and text search results.

        Args:
            query_embedding: The query embedding vector.
            query_text: The raw query text for full-text search.
            top_k: Number of results to return.

        Returns:
            Fused results ordered by combined RRF score.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH vector_search AS (
                    SELECT id, content, source,
                           ROW_NUMBER() OVER (
                               ORDER BY embedding <=> $1::vector
                           ) AS vector_rank
                    FROM documents
                    ORDER BY embedding <=> $1::vector
                    LIMIT 20
                ),
                text_search AS (
                    SELECT id, content, source,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank_cd(
                                   content_tsv,
                                   websearch_to_tsquery('english', $2)
                               ) DESC
                           ) AS text_rank
                    FROM documents
                    WHERE content_tsv @@ websearch_to_tsquery('english', $2)
                    LIMIT 20
                )
                SELECT
                    COALESCE(v.id, t.id) AS id,
                    COALESCE(v.content, t.content) AS content,
                    COALESCE(v.source, t.source) AS source,
                    COALESCE(1.0 / (60 + v.vector_rank), 0) +
                    COALESCE(1.0 / (60 + t.text_rank), 0) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN text_search t ON v.id = t.id
                ORDER BY rrf_score DESC
                LIMIT $3
                """,
                str(query_embedding), query_text, top_k,
            )

        return [dict(row) for row in rows]

# Usage
async def main():
    store = await PgVectorStore.create(
        "postgresql://user:password@localhost:5432/ragdb"
    )

    # Get embedding for a query
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="How do vector databases work?"
    )
    query_embedding = response.data[0].embedding

    # Pure vector search
    results = await store.search(query_embedding, top_k=5)

    # Hybrid search (vector + full-text)
    results = await store.hybrid_search(
        query_embedding,
        query_text="vector databases architecture",
        top_k=5
    )

    for r in results:
        print(f"Score: {r.get('rrf_score', r.get('similarity')):.4f} | "
              f"{r['content'][:100]}...")
```

### Qdrant vs pgvector Decision Framework

Choosing between a purpose-built vector database and a PostgreSQL extension depends on your scale, operational constraints, and feature requirements.

| Criterion | Qdrant | pgvector |
|-----------|--------|----------|
| **Search speed** | Faster (purpose-built engine, HNSW optimized) | Slower at scale (shared resources with RDBMS) |
| **Filtering** | Powerful payload filters (nested, geo, range) | SQL WHERE clauses (more expressive for complex joins) |
| **Scaling** | Distributed mode, automatic sharding | PostgreSQL replication, partitioning |
| **Quantization** | Built-in (scalar INT8, product) | None (requires external compression) |
| **Hybrid search** | Native sparse vectors | tsvector + vector (SQL-based fusion) |
| **Operations** | Separate service to deploy and monitor | Part of existing PostgreSQL infrastructure |
| **Ecosystem** | REST/gRPC API, language clients | SQL interface, ORM integration |
| **Data consistency** | Eventually consistent (by default) | Full ACID transactions |
| **When to choose** | >1M vectors, high throughput, dedicated search | <500K vectors, existing PostgreSQL, ACID needed |

**Decision heuristic:**

- If you already run PostgreSQL and your corpus is under 500K documents, start with pgvector. You avoid the operational overhead of a separate service, get ACID transactions, and can run hybrid queries that join vector results with relational data.
- If you need to scale beyond 1M vectors, require sub-10ms latency, or want built-in quantization and distributed search, deploy Qdrant. The operational cost pays for itself in performance and features.

---

## Semantic Caching

LLM inference is expensive. When users ask semantically similar questions, a semantic cache can return a previous answer instead of making a new LLM call — saving both latency and cost.

Unlike traditional exact-match caches, a semantic cache uses embedding similarity to detect questions that are different in wording but identical in intent.

```python
import hashlib
import json
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, Range,
)

class SemanticCache:
    """LLM response cache using semantic similarity.

    Stores question-answer pairs with their embeddings. When a new
    question arrives, it checks whether a semantically similar question
    has been answered before. If the similarity exceeds a threshold,
    the cached answer is returned without calling the LLM.
    """

    def __init__(
        self,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "semantic_cache",
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ):
        """Initialize the semantic cache.

        Args:
            qdrant_url: Qdrant server hostname.
            qdrant_port: Qdrant server port.
            collection: Name of the cache collection.
            similarity_threshold: Minimum cosine similarity to consider
                a cache hit.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self.client = QdrantClient(qdrant_url, port=qdrant_port)
        self.openai = OpenAI()
        self.collection = collection
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the cache collection if it does not already exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection for c in collections):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

    def _embed(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def get(self, question: str) -> str | None:
        """Look up a semantically similar question in the cache.

        Args:
            question: The user's question.

        Returns:
            The cached answer if a sufficiently similar question was
            found and the entry has not expired; None otherwise.
        """
        embedding = self._embed(question)
        now = time.time()

        results = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="expires_at",
                        range=Range(gte=now)  # Only non-expired entries
                    )
                ]
            ),
            limit=1
        )

        if results and results[0].score >= self.similarity_threshold:
            return results[0].payload["answer"]

        return None

    def set(self, question: str, answer: str):
        """Store a question-answer pair in the cache.

        Args:
            question: The question that was asked.
            answer: The answer that was generated.
        """
        embedding = self._embed(question)
        point_id = hashlib.md5(question.encode()).hexdigest()

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question": question,
                        "answer": answer,
                        "created_at": time.time(),
                        "expires_at": time.time() + self.ttl_seconds,
                    }
                )
            ]
        )

# Usage with an LLM call
cache = SemanticCache(similarity_threshold=0.95, ttl_seconds=7200)

def cached_llm_call(question: str) -> str:
    """LLM call with semantic caching.

    Checks the semantic cache before calling the LLM. If a similar
    question was answered recently, returns the cached response.

    Args:
        question: The user's question.

    Returns:
        The answer, either from cache or freshly generated.
    """
    # Check cache first
    cached = cache.get(question)
    if cached:
        print("[CACHE HIT]")
        return cached

    # Cache miss — call the LLM
    print("[CACHE MISS]")
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    answer = response.choices[0].message.content

    # Store in cache for next time
    cache.set(question, answer)
    return answer

# These two calls are semantically identical but worded differently.
# The second one should hit the cache.
answer1 = cached_llm_call("What is the capital of France?")
answer2 = cached_llm_call("Which city is the capital of France?")
```

**Tuning the similarity threshold:** A threshold of 0.95 is conservative — it requires near-identical questions. Lowering to 0.90 increases cache hit rate but risks returning answers to slightly different questions. Start conservative and lower the threshold based on observed false-positive rates.

---

## Key Insights

> **Production RAG at 1M+ documents:** Follow this stack for reliable, scalable RAG: (1) **Chunking:** semantic or recursive, 500-1000 tokens with 10-20% overlap. (2) **Embeddings:** `text-embedding-3-small` for speed or `text-embedding-3-large` for quality. (3) **Vector DB:** Qdrant with scalar quantization. (4) **Retrieval:** hybrid dense + BM25 with Reciprocal Rank Fusion. (5) **Re-ranking:** cross-encoder, retrieve top-20, rerank to top-5. (6) **Monitoring:** RAGAS metrics on a weekly cadence, user feedback loop for continuous improvement.

> **Evaluating RAG Quality:** The RAGAS framework provides four automated metrics: `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`. These should be supplemented with A/B testing against live traffic, periodic human evaluation (especially in regulated domains), and domain-specific metrics such as citation accuracy or compliance verification.

---

## References

- Qdrant Documentation: [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)
- pgvector — Open-source vector similarity search for PostgreSQL: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- RAGAS — Evaluation framework for RAG: [https://docs.ragas.io/](https://docs.ragas.io/)
- *Retrieval-Augmented Generation for Large Language Models: A Survey*: [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
- LangChain RAG Tutorial: [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)
