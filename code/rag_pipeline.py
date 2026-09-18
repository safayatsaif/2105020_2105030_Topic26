"""
rag_pipeline.py — RAG Pipeline Implementation
Implemented by: Tanvin Islam Abir (2105020)

This module implements the complete Retrieval-Augmented Generation pipeline:
  1. Document ingestion and chunking
  2. Embedding via sentence-transformers (all-MiniLM-L6-v2)
  3. Vector storage via ChromaDB
  4. Similarity-based retrieval
  5. Prompt assembly
  6. LLM generation via Ollama
"""

import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
import ollama as ollama_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DIR       = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME  = "acme_knowledge_base"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL        = "llama3"          # change to "mistral" if preferred
CHUNK_SIZE       = 500               # characters per chunk
CHUNK_OVERLAP    = 50                # overlap between consecutive chunks
TOP_K            = 3                 # number of chunks to retrieve

# System prompt used by the RAG orchestrator
SYSTEM_PROMPT = (
    "You are a helpful customer service assistant for AcmeCorp. "
    "Answer questions based ONLY on the provided context. "
    "Do not make up information. If the context does not contain the answer, "
    "say you don't know."
)


# ---------------------------------------------------------------------------
# Embedding wrapper
# ---------------------------------------------------------------------------
class Embedder:
    """Wraps sentence-transformers to produce 384-dim dense vectors."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        print(f"[RAG] Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print(f"[RAG] Embedding model loaded (dim={self.model.get_sentence_embedding_dimension()}).")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return list of embedding vectors for the given texts."""
        vectors = self.model.encode(texts, show_progress_bar=False)
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Document chunker
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a document into overlapping chunks of approximately `chunk_size`
    characters. Splitting is done at sentence boundaries when possible.
    """
    # Simple sentence-boundary splitting
    sentences = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        # Split on period-space as a rough sentence boundary
        parts = line.replace(". ", ".\n").split("\n")
        sentences.extend(p.strip() for p in parts if p.strip())

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of the previous chunk
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


# ---------------------------------------------------------------------------
# Vector Store (ChromaDB wrapper)
# ---------------------------------------------------------------------------
class VectorStore:
    """Persistent ChromaDB collection for document embeddings."""

    def __init__(self, embedder: Embedder, persist_dir: str = CHROMA_DIR,
                 collection_name: str = COLLECTION_NAME):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[RAG] ChromaDB collection '{collection_name}' ready "
              f"({self.collection.count()} existing chunks).")

    def add_document(self, doc_text: str, source: str = "unknown") -> int:
        """Chunk a document, embed it, and store in ChromaDB. Returns chunk count."""
        chunks = chunk_text(doc_text)
        if not chunks:
            return 0

        ids = []
        for i, chunk in enumerate(chunks):
            # Deterministic ID based on content hash
            h = hashlib.sha256(chunk.encode()).hexdigest()[:16]
            ids.append(f"{source}_{i}_{h}")

        embeddings = self.embedder.embed(chunks)
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[RAG] Ingested {len(chunks)} chunks from source '{source}'.")
        return len(chunks)

    def add_raw_chunk(self, chunk_text_str: str, doc_id: str,
                      source: str = "adversarial") -> None:
        """Add a single pre-crafted chunk directly (used by the attacker)."""
        embedding = self.embedder.embed([chunk_text_str])
        self.collection.upsert(
            ids=[doc_id],
            documents=[chunk_text_str],
            embeddings=embedding,
            metadatas=[{"source": source}],
        )
        print(f"[RAG] Injected raw chunk '{doc_id}' into vector store.")

    def query(self, query_text: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the top-k most similar chunks for a query."""
        query_embedding = self.embedder.embed([query_text])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id":       results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return retrieved

    def reset(self) -> None:
        """Delete all documents from the collection."""
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
        print("[RAG] Vector store reset.")


# ---------------------------------------------------------------------------
# Prompt assembler
# ---------------------------------------------------------------------------
def assemble_prompt(system_prompt: str, context_chunks: list[str],
                    user_query: str) -> list[dict]:
    """
    Build the chat-format prompt for Ollama.
    Returns a list of message dicts: [system, user].
    """
    context_block = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )
    user_content = (
        f"Context from knowledge base:\n\n{context_block}\n\n"
        f"Based on the above context, answer the following question:\n"
        f"\"{user_query}\""
    )
    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM generation via Ollama
# ---------------------------------------------------------------------------
def generate_response(messages: list[dict], model: str = LLM_MODEL) -> str:
    """Send the assembled prompt to Ollama and return the response text."""
    try:
        response = ollama_client.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        return f"[LLM Error] {e}"


# ---------------------------------------------------------------------------
# Full RAG query (end-to-end)
# ---------------------------------------------------------------------------
class RAGPipeline:
    """End-to-end RAG pipeline tying everything together."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT,
                 llm_model: str = LLM_MODEL):
        self.embedder = Embedder()
        self.store = VectorStore(self.embedder)
        self.system_prompt = system_prompt
        self.llm_model = llm_model

    def ingest(self, doc_text: str, source: str = "doc") -> int:
        """Ingest a document into the knowledge base."""
        return self.store.add_document(doc_text, source)

    def inject_chunk(self, chunk: str, doc_id: str) -> None:
        """Inject a single raw chunk (attacker interface)."""
        self.store.add_raw_chunk(chunk, doc_id)

    def query(self, user_query: str, top_k: int = TOP_K,
              return_context: bool = False) -> dict:
        """
        Run a full RAG query:
          1. Retrieve top-k chunks
          2. Assemble prompt
          3. Generate LLM response
        Returns a dict with 'response', 'retrieved_chunks', and 'prompt'.
        """
        retrieved = self.store.query(user_query, top_k)
        context_chunks = [r["document"] for r in retrieved]
        messages = assemble_prompt(self.system_prompt, context_chunks, user_query)
        response = generate_response(messages, self.llm_model)
        result = {
            "response": response,
            "retrieved_chunks": retrieved,
        }
        if return_context:
            result["prompt_messages"] = messages
        return result

    def reset(self) -> None:
        """Reset the knowledge base."""
        self.store.reset()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline — Standalone Test")
    print("=" * 60)

    # Sample AcmeCorp knowledge base documents
    docs = [
        (
            "AcmeCorp offers a 30-day refund policy for all products purchased "
            "online. Customers must provide proof of purchase and return the "
            "item in its original packaging. Refunds are processed within 5-7 "
            "business days after the returned item is received at our warehouse. "
            "Shipping costs for returns are non-refundable unless the item "
            "arrived damaged or defective."
        ),
        (
            "For returns, contact support@acmecorp.com or call 1-800-ACME "
            "within 30 days of delivery. Our customer service team is available "
            "Monday through Friday, 9 AM to 6 PM EST. For international orders, "
            "return shipping is the responsibility of the customer."
        ),
        (
            "AcmeCorp's premium membership program offers free shipping on all "
            "orders, early access to new products, and exclusive discounts of up "
            "to 20%. Premium members also get priority customer support with "
            "dedicated phone and email channels."
        ),
        (
            "AcmeCorp was founded in 2010 and is headquartered in San Francisco, "
            "California. The company specializes in consumer electronics and "
            "smart home devices. As of 2025, AcmeCorp serves over 2 million "
            "customers across 15 countries."
        ),
    ]

    pipeline = RAGPipeline()
    pipeline.reset()

    print("\n--- Ingesting documents ---")
    for i, doc in enumerate(docs):
        pipeline.ingest(doc, source=f"acme_doc_{i}")

    print("\n--- Running test query ---")
    test_query = "What is the refund policy?"
    result = pipeline.query(test_query, return_context=True)
    print(f"\nQuery: {test_query}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"]):
        print(f"  [{i+1}] (dist={chunk['distance']:.4f}) {chunk['document'][:80]}...")
    print(f"\nLLM Response:\n{result['response']}")
