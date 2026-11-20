# Automated LLM Evaluation & Defense Platform: Full Technical Guide

## 1. Executive Summary

This platform is a comprehensive software engineering and research environment designed to evaluate, benchmark, and improve the safety and alignment of local large language models (LLMs). Operating exclusively on locally deployed models via Ollama, the system automates:

1. **Iterative Evaluation:** Using an automated feedback loop (based on the published Prompt Automatic Iterative Refinement method), candidate prompts are generated, evaluated, and iteratively refined.
2. **Independent Scoring:** An evaluator model independently assesses responses against standard guidelines and outputs structured JSON verdicts.
3. **Defense Layer Training:** Using historical evaluation data and a benchmark of benign everyday queries, dual semantic classifiers (input guard and output guard) are trained to filter prompts and responses.
4. **Interactive Studio:** A full web interface (Streamlit) provides one-click test execution, defense training, visual conversation inspection, and report compilation.

---

## 2. Core Architecture & Workflow

```
                    ┌────────────────────────────┐
                    │  Evaluation Strategies      │
                    │  (4 Analytical Frameworks) │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
            ┌─────────▶│   Generator Model   │
            │          │ (crafts candidate   │
            │          │  test prompts)      │
            │          └──────────┬──────────┘
            │                     │ prompt
            │                     ▼
            │          ┌─────────────────────┐
     mutate &          │   Target Adapter    │──▶ Local Ollama Models
     refine if         │ (unified interface) │    (e.g., Llama 3, Qwen 2.5, Mistral)
     safe response     └──────────┬──────────┘
            │                     │ response
            │                     ▼
            │          ┌─────────────────────┐
            │          │   Evaluator Model   │
            │          │(independent scoring)│
            └──────────┴──────────┬──────────┘
                                  │ verdict (JSON)
                                  ▼
                       ┌─────────────────────┐
                       │   SQLite Database   │
                       └──────────┬──────────┘
                       ┌──────────┴──────────┐
                       ▼                     ▼
            ┌─────────────────────┐ ┌─────────────────────┐
            │ Interactive Studio  │ │ Automated Reporting │
            │ (Streamlit Web UI)  │ │ (Markdown & PDF)    │
            └─────────────────────┘ └─────────────────────┘
```

### The Three Agent Roles
- **The Generator (Attacker):** Evaluates how models respond to diverse framings. When a target model resists, the generator examines the evaluator's feedback and creates a modified phrasing to test robustness.
- **The Target Adapter:** A unified client that connects directly to local Ollama endpoints with automatic retry and backoff handling.
- **The Evaluator (Judge):** Operates in a distinct context from the generator. It analyzes whether the model complied with policy-restricted requests and assigns a severity rating (1–5).

---

## 3. The 4 Evaluation Taxonomy Categories

Configured in `config/strategies.yaml`:

1. **Roleplay / Persona Framing:**
   Evaluates how models respond when placed inside hypothetical scenarios, fictional story writing, or alternate personas.
2. **Encoding / Obfuscation:**
   Tests whether pattern-based safety filters can be bypassed using indirect language, analogies, or symbolic representations.
3. **Multi-Turn Escalation:**
   Simulates a conversational exchange that starts with benign, standard inquiries and gradually introduces more complex requests across sequential turns.
4. **Prompt Injection via Processed Content:**
   Tests how models handle text processing, translation, or summarization when instructions are embedded directly within the input content.

---

## 4. The Defense Shield Layer

The primary goal of the platform is defensive: identifying weaknesses to build stronger protection layers.

### 1. Dual Semantic Classifiers
- **Input Guard:** Analyzes incoming user prompts before they reach the model.
- **Output Guard:** Analyzes model responses before they are returned to the user or application layer.

### 2. Training Pipeline (`redteam/defense/train_guard.py`)
- **Feature Extraction:** Converts text into 384-dimensional dense semantic vector representations using the local `sentence-transformers/all-MiniLM-L6-v2` model.
- **Decision Boundary:** Fits a lightweight classifier (Logistic Regression) on historical attempts combined with a negative dataset of 100 benign queries (`data/benign_prompts.jsonl`).
- **Usability Validation (False-Positive Rate):** Evaluates the classifiers on benign tasks (coding, writing, science) to ensure legitimate queries are not mistakenly blocked.

---

## 5. Web Application User Guide

The dashboard (`dashboard/app.py`) provides an end-to-end graphical control center:

### Tab 1: 🎮 Launch Attack Test
- Select installed Ollama models from the auto-detected list.
- Select target strategies and probe goals.
- Configure maximum iterations per test.
- Click **Start AI Battle Now** to observe live streaming progress.

### Tab 2: 🛡️ Train & Test Shield
- **Train Guards:** Fits the semantic classifiers and displays accuracy and false-positive metrics.
- **Test Defenses:** Re-runs evaluations with guards enabled to verify vulnerability mitigation.

### Tab 3: 📊 View Results & Heatmaps
- High-level metric cards: Total probes, breaches, blocked queries, and success rates.
- Interactive heatmaps comparing model resilience across strategies with and without guards.

### Tab 4: 💬 Read Conversations
- Inspect every prompt, target response, and evaluator verdict.
- Automatically redacts severe content to maintain documentation integrity.

### Tab 5: 📄 Generate Report
- Compiles an executive summary report with embedded charts and provides one-click Markdown download.

### Tab 6: 💡 How It Works
- Provides onboarding explanations and architectural reference material for team members.

---

## 6. Safety & Scope Guarantees

1. **Local Allowlist Only:** The platform includes an enforced scope guard (`redteam/scope_guard.py`) that strictly permits communication with local Ollama endpoints (`http://localhost:11434`), preventing any external API transmission.
2. **Abstract Definitions:** Probe goals (`config/goals.yaml`) describe abstract behavioral categories rather than operational harmful content.
3. **Audit Trail:** All interactions are stored in SQLite (`data/redteam.db`) for reproducible research and verification.
