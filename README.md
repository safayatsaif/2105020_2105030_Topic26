# CSE 406: Computer Security — Topic 26
## Indirect Prompt Injection via RAG

**Group Members:**
- Tanvin Islam Abir (2105020)
- Safayat Saif (2105030)

**Semester:** January 2026

---

## Folder Structure

```
2105020_2105030_Topic26/
├── README.md                  ← This file
├── COMPLETE_PROJECT_EXPLANATION.html ← Comprehensive 9-section project & paper breakdown
├── attack_architecture.jpg    ← Architecture diagram
├── defense_layers.jpg         ← Multi-layer defense diagram
├── payload_structure.jpg      ← Bi-zone adversarial document layout
├── code/                      ← All implementation source code
│   ├── rag_pipeline.py        ← RAG pipeline (embedding, ChromaDB, retrieval, LLM, pinned seed=42)
│   ├── payload_crafter.py     ← Adversarial payload crafting (bi-zone, markdown exfiltration)
│   ├── c2_server.py           ← Attacker's C2 exfiltration server + real-time dashboard
│   ├── victim_app.py          ← Victim chat web application (Flask, live defense toggle, REST injection)
│   ├── defense.py             ← Defense mechanisms (Spotlighting, CSP, Output Sanitization)
│   ├── run_attack.py          ← End-to-end attack runner (6 steps, 100% success rate)
│   ├── run_demo.py            ← Full demo (Phase A undefended vs. Phase B defended comparison)
│   ├── requirements.txt       ← Python dependencies
│   └── templates/
│       └── chat.html          ← Chat UI template with real-time defense toggle
└── reports/
    ├── CSE406_Design_Report_Topic26_Indirect_Prompt_Injection_RAG.html   ← Design report (Week 11)
    ├── Design_Report_Indirect_Prompt_Injection_RAG.pdf                   ← Submitted Design report PDF
    ├── CSE406_Final_Report_Topic26_Indirect_Prompt_Injection_RAG.html    ← Final report (Week 13-14)
    ├── Final_Report_Indirect_Prompt_Injection_RAG.pdf                    ← Submitted Final report PDF
    ├── topology_diagram_1789190256815.jpg                                ← Topology diagram
    ├── snapshot_victim_attacked.png                                      ← Victim screen + network exfiltration
    ├── snapshot_c2_dashboard.png                                         ← C2 exfiltration dashboard screen
    └── snapshot_defense_blocked.png                                      ← Defended client + CSP block console
```

---

## Assignment Requirements Compliance & Pinned Parameters

As mandated by the CSE 406 Project Specification for AI/LLM attack tools:
- **Zero Third-Party Attack Libraries:** Custom payload generator, custom C2 listener, custom defense module. No foolbox, ART, or TextAttack used.
- **Pinned Seed:** `RANDOM_SEED = 42` across Python `random`, `numpy.random`, and `torch.manual_seed`.
- **Pinned Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Pinned LLM:** `llama3:latest` running locally via Ollama.
- **Pinned Dataset:** AcmeCorp corporate support corpus (5 documents, chunked at 500 chars with 50 char overlap).
- **Offline / Standalone Reproducibility:** Both `run_demo.py` and `run_attack.py` feature `--mock` mode, guaranteeing that any grader can verify the end-to-end pipeline deterministically even without an active GPU/Ollama instance.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running → [https://ollama.ai](https://ollama.ai) (optional if using `--mock`)
3. Pull an LLM model:
   ```bash
   ollama pull llama3
   ```

## Setup

```bash
cd code
pip install -r requirements.txt
```

## How to Run

### Option 1: Quick Terminal Demo (Attack + Defense Comparison)
```bash
cd code
# Using live Ollama:
python run_demo.py --model llama3

# Or using deterministic simulation mode (instant, no Ollama needed):
python run_demo.py --mock
```

### Option 2: Full Web-Based Demo (3 Terminals)

**Terminal 1 — Start C2 server (attacker):**
```bash
python c2_server.py
```

**Terminal 2 — Start victim chat app (undefended):**
```bash
python victim_app.py --model llama3
```

**Terminal 3 — Inject attack & test:**
```bash
python run_attack.py --model llama3
```

Then open [http://localhost:5000](http://localhost:5000) and ask "What is the refund policy?"
Check [http://localhost:9999](http://localhost:9999) for the C2 dashboard with stolen data.

### Option 3: Show Defense Working
```bash
python victim_app.py --defense --port 5001 --model llama3
```
Open [http://localhost:5001](http://localhost:5001) — this version blocks the attack.

---

## Member Responsibilities

| Component | Abir (2105020) | Saif (2105030) |
|---|---|---|
| RAG Pipeline (`rag_pipeline.py`) | Lead | Support |
| Payload Crafter (`payload_crafter.py`) | Support | Lead |
| C2 Server (`c2_server.py`) | Lead | — |
| Victim App (`victim_app.py` + `chat.html`) | — | Lead |
| Defense (`defense.py`) | Support | Lead |
| Attack/Demo Runners | Lead | Support |
| Report: Attack Steps & Outputs | Lead | Support |
| Report: Success Analysis & Defense | Support | Lead |
