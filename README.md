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
├── code/                      ← All implementation source code
│   ├── rag_pipeline.py        ← RAG pipeline (embedding, ChromaDB, retrieval, LLM)
│   ├── payload_crafter.py     ← Adversarial payload crafting
│   ├── c2_server.py           ← Attacker's C2 exfiltration server
│   ├── victim_app.py          ← Victim chat web application (Flask)
│   ├── defense.py             ← Defense mechanisms (spotlighting, CSP, sanitization)
│   ├── run_attack.py          ← End-to-end attack runner
│   ├── run_demo.py            ← Full demo (attack + defense comparison)
│   ├── requirements.txt       ← Python dependencies
│   └── templates/
│       └── chat.html          ← Chat UI template
└── reports/
    ├── CSE406_Design_Report_Topic26_Indirect_Prompt_Injection_RAG.html   ← Design report
    ├── CSE406_Final_Report_Topic26_Indirect_Prompt_Injection_RAG.html    ← Final report
    ├── Design_Report_Indirect_Prompt_Injection_RAG.pdf                   ← Design report (PDF)
    └── topology_diagram_1789190256815.jpg                                ← Topology diagram
```

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running → [https://ollama.ai](https://ollama.ai)
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
python run_demo.py --model llama3
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
