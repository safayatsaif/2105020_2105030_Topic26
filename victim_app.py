"""
victim_app.py — Victim Web Chat Application (Flask)
Implemented by: Safayat Saif (2105030)

This module runs the victim-facing RAG chat application. It provides:
  1. A web chat interface where users ask questions
  2. Backend RAG pipeline integration (retrieval + LLM generation)
  3. Toggle between undefended mode (vulnerable) and defended mode
  4. Markdown rendering in the UI (intentionally permissive in undefended mode)

In undefended mode, the app renders LLM output including markdown images
directly in the browser, enabling the exfiltration side-channel attack.

In defended mode, the app applies:
  - Spotlighting on retrieved chunks
  - CSP headers blocking external image loads
  - Output sanitization stripping external URLs
"""

import os
import sys
import argparse
from flask import Flask, render_template, request, jsonify

# Import our modules
from rag_pipeline import RAGPipeline, SYSTEM_PROMPT, assemble_prompt, generate_response
from defense import DefensePipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VICTIM_PORT = 5000
VICTIM_HOST = "0.0.0.0"


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")

# Global state (set during init)
rag_pipeline = None
defense_pipeline = None
defense_enabled = False


def init_app(enable_defense: bool = False, llm_model: str = "llama3"):
    """Initialize the RAG pipeline and optionally enable defenses."""
    global rag_pipeline, defense_pipeline, defense_enabled

    rag_pipeline = RAGPipeline(llm_model=llm_model)
    defense_enabled = enable_defense

    if rag_pipeline.store.collection.count() == 0:
        print("[VICTIM APP] Knowledge base is empty. Seeding initial documents...")
        from run_attack import LEGITIMATE_DOCS
        for doc in LEGITIMATE_DOCS:
            rag_pipeline.ingest(doc["text"], source=doc["source"])
        print(f"[VICTIM APP] Seeded {rag_pipeline.store.collection.count()} chunks.")

    if enable_defense:
        defense_pipeline = DefensePipeline(
            enable_spotlighting=True,
            enable_csp=True,
            enable_sanitization=True,
        )
        print("[VICTIM APP] Defense mechanisms ENABLED")
        print(f"[VICTIM APP] Defense status: {defense_pipeline.status()}")
    else:
        defense_pipeline = None
        print("[VICTIM APP] Running WITHOUT defenses (vulnerable mode)")


@app.route("/")
def index():
    """Serve the chat UI."""
    response = app.make_response(
        render_template("chat.html", defense_enabled=defense_enabled)
    )

    # Apply CSP headers if defense is enabled
    if defense_enabled and defense_pipeline:
        csp_header = defense_pipeline.get_csp_header()
        response.headers["Content-Security-Policy"] = csp_header
        print(f"[VICTIM APP] CSP header applied: {csp_header}")

    return response


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat query from the victim user."""
    global rag_pipeline
    try:
        data = request.get_json(silent=True) or {}
        user_query = str(data.get("query", "")).strip()

        if not user_query:
            return jsonify({"error": "Empty query", "response": "Please enter a question."}), 400

        print(f"\n[VICTIM APP] Received query: {user_query}")

        if rag_pipeline is None:
            init_app(enable_defense=defense_enabled)

        # Step 1: Retrieve context chunks from the vector store
        retrieved = rag_pipeline.store.query(user_query) or []
        valid_retrieved = []
        context_chunks = []
        for r in retrieved:
            doc_text = r.get("document") if isinstance(r, dict) else None
            if doc_text and isinstance(doc_text, str):
                valid_retrieved.append(r)
                context_chunks.append(doc_text)

        print(f"[VICTIM APP] Retrieved {len(valid_retrieved)} valid chunks")
        for i, r in enumerate(valid_retrieved):
            preview = str(r.get("document", ""))[:80].replace("\n", " ")
            dist = r.get("distance", 0.0)
            print(f"  Chunk {i+1} (dist={dist:.4f}): {preview}...")

        # Step 2: Apply input-side defenses (spotlighting)
        system_prompt = SYSTEM_PROMPT
        defense_info = {}

        if defense_enabled and defense_pipeline:
            system_prompt, context_chunks = defense_pipeline.defend_input(
                system_prompt, context_chunks
            )
            defense_info["spotlighting"] = True
            defense_info["nonce"] = defense_pipeline._current_nonce
            print(f"[VICTIM APP] Spotlighting applied (nonce={defense_pipeline._current_nonce})")

        # Step 3: Assemble prompt and generate LLM response
        messages = assemble_prompt(system_prompt, context_chunks, user_query)
        llm_response = generate_response(messages, rag_pipeline.llm_model)

        print(f"[VICTIM APP] LLM response length: {len(llm_response)} chars")

        # Step 4: Apply output-side defenses (sanitization)
        threats_detected = 0
        stripped_urls = []

        if defense_enabled and defense_pipeline:
            sanitize_result = defense_pipeline.defend_output(llm_response)
            llm_response = sanitize_result["sanitized_text"]
            threats_detected = sanitize_result["threats_detected"]
            stripped_urls = sanitize_result["stripped_urls"]
            defense_info["sanitization"] = True
            if threats_detected > 0:
                print(f"[VICTIM APP] *** THREATS BLOCKED: {threats_detected} external URLs stripped ***")
                for url in stripped_urls:
                    print(f"  Blocked: {url}")
        else:
            # Check if the response contains suspicious content (for logging only)
            if "![" in llm_response and "http" in llm_response:
                print(f"[VICTIM APP] *** WARNING: Response contains external image URL "
                      f"(NO DEFENSE ACTIVE — data may be exfiltrated) ***")

        # Build response
        result = {
            "response": llm_response,
            "retrieved_chunks": [
                {
                    "document": str(r.get("document", ""))[:100] + "...",
                    "distance": r.get("distance", 0.0),
                    "source": (r.get("metadata") or {}).get("source", "unknown"),
                }
                for r in valid_retrieved
            ],
        }

        if defense_enabled:
            result["defense_applied"] = defense_info
            result["threats_detected"] = threats_detected
            if stripped_urls:
                result["stripped_urls"] = stripped_urls

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"[VICTIM APP] Exception in /chat: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "response": f"Server encountered an error while processing: {e}"
        }), 500


@app.route("/status")
def status():
    """Return current app status."""
    return jsonify({
        "defense_enabled": defense_enabled,
        "defense_status": defense_pipeline.status() if defense_pipeline else None,
        "knowledge_base_size": rag_pipeline.store.collection.count(),
        "llm_model": rag_pipeline.llm_model,
    })


@app.route("/toggle_defense", methods=["POST", "GET"])
def toggle_defense():
    """Toggle defense dynamically without restarting the server."""
    global defense_enabled, defense_pipeline
    defense_enabled = not defense_enabled
    if defense_enabled:
        defense_pipeline = DefensePipeline(
            enable_spotlighting=True,
            enable_csp=True,
            enable_sanitization=True,
        )
        print("[VICTIM APP] Defense mechanisms ENABLED dynamically via API/UI.")
    else:
        defense_pipeline = None
        print("[VICTIM APP] Defense mechanisms DISABLED dynamically via API/UI.")
    return jsonify({
        "defense_enabled": defense_enabled,
        "status": "DEFENSE ON" if defense_enabled else "NO DEFENSE"
    })


@app.route("/inject", methods=["POST"])
def inject():
    """Endpoint for attacker to inject an adversarial chunk into the live vector store."""
    data = request.get_json() or {}
    chunk = data.get("chunk") or data.get("payload")
    doc_id = data.get("doc_id", "remote_injection_001")
    if not chunk:
        return jsonify({"error": "Missing 'chunk' or 'payload' field"}), 400

    rag_pipeline.inject_chunk(chunk, doc_id=doc_id)
    print(f"[VICTIM APP] Remote adversarial chunk '{doc_id}' injected successfully.")
    return jsonify({
        "status": "success",
        "doc_id": doc_id,
        "total_chunks": rag_pipeline.store.collection.count()
    })


@app.route("/reset", methods=["POST", "GET"])
def reset():
    """Reset the knowledge base back to legitimate state."""
    rag_pipeline.reset()
    from run_attack import LEGITIMATE_DOCS
    for doc in LEGITIMATE_DOCS:
        rag_pipeline.ingest(doc["text"], source=doc["source"])
    print(f"[VICTIM APP] Knowledge base reset and reseeded ({rag_pipeline.store.collection.count()} chunks).")
    return jsonify({
        "status": "reset",
        "total_chunks": rag_pipeline.store.collection.count()
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcmeCorp RAG Chat — Victim Application")
    parser.add_argument("--defense", action="store_true",
                        help="Enable defense mechanisms (spotlighting + CSP + sanitization)")
    parser.add_argument("--port", type=int, default=VICTIM_PORT,
                        help=f"Port to run on (default: {VICTIM_PORT})")
    parser.add_argument("--model", type=str, default="llama3",
                        help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    import socket
    def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    if args.port == 5000 and is_port_in_use(5000):
        print("[VICTIM APP] Port 5000 is occupied (commonly by macOS AirPlay Receiver). Switching to port 5002.")
        args.port = 5002

    init_app(enable_defense=args.defense, llm_model=args.model)

    print(f"\n[VICTIM APP] Starting on http://localhost:{args.port}")
    print(f"[VICTIM APP] Defense: {'ENABLED' if args.defense else 'DISABLED'}")
    print(f"[VICTIM APP] LLM Model: {args.model}\n")

    app.run(host=VICTIM_HOST, port=args.port, debug=False)
