# SynapseAML — XGBoost Fraud Detection with SHAP Explainability and Multi-Agent AML Investigation Pipeline

> Trained an XGBoost classifier for financial transaction anomaly detection, with SHAP `TreeExplainer` for per-prediction feature attribution, and a CrewAI multi-agent system for automated compliance validation, report generation, and case management.

---

## Why This Project

Money laundering is estimated to account for 2–5% of global GDP. Detecting it is a hard ML problem: the data is highly imbalanced, the cost of a false negative (missed fraud) far exceeds the cost of a false positive, and a raw prediction score is not enough — compliance teams need to know *why* a transaction was flagged, and that explanation needs to trigger an investigation workflow automatically.

SynapseAML addresses all three layers:
1. **A trained XGBoost model** that predicts whether a transaction is suspicious, optimized for recall on imbalanced data
2. **SHAP `TreeExplainer`** that generates exact per-feature attributions for every prediction — making the model auditable and its decisions explainable to compliance officers
3. **A CrewAI multi-agent pipeline** that takes the model's output and automates the downstream workflow — entity screening, investigation report generation, and case creation — end to end

---

## ML Model

### Problem
Binary classification on imbalanced financial transaction data: **Normal (0) vs. Suspicious (1)**.

### Why XGBoost
- Strong benchmark performance on tabular financial data
- Native handling of missing values
- L1/L2 regularization prevents overfitting on imbalanced classes
- Fully compatible with SHAP `TreeExplainer` for fast, exact Shapley values
- Easily serializable via `joblib` for production inference

### Pipeline
XGBoost is wrapped in a scikit-learn `Pipeline` with one-hot encoding for categorical features, ensuring no data leakage between train/test splits. Saved as `src/model.joblib`.

### Features

| Feature | Type | Description |
| --- | --- | --- |
| From Bank | Categorical | Sender institution code |
| To Bank | Categorical | Receiver institution code |
| Account / Account Dest | Categorical | Sender and receiver account IDs |
| Amount Received / Paid | Numerical | Transaction amounts |
| Receiving / Payment Currency | Categorical | Source and destination currencies |
| Payment Format | Categorical | Transfer method (Wire, Cheque, etc.) |
| Date, Day, Time | Numerical/Categorical | Temporal transaction features |

### Model Performance

The dataset is highly imbalanced (~99% Normal, ~1% Suspicious). `RandomUnderSampler` was applied before training to produce a balanced training set — so accuracy alone is a misleading metric here. **F1 Score and ROC-AUC are the primary evaluation criteria**.

| Metric | Score |
| --- | --- |
| **F1 Score** | **91.08%** |
| **ROC-AUC** | **95.58%** |
| Recall (Suspicious class) | 96.25% |
| Precision (Suspicious class) | 86.44% |
| Accuracy | 90.78% |

The model achieves **96.25% recall on the Suspicious class** — catching 96 out of every 100 money laundering transactions — with a **95.58% ROC-AUC** confirming strong discrimination ability independent of the classification threshold.

---

## Explainability — SHAP TreeExplainer

A model score alone is not sufficient in a compliance context. Regulators and investigators need to understand *which specific transaction attributes* triggered a flag and *how much* each one contributed — not just that the model said suspicious.

### Why SHAP
SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory: it assigns each feature a contribution value based on its marginal impact across all possible feature orderings. Unlike simple feature importance (which is global and aggregate), SHAP produces **local, per-prediction attributions** — meaning every flagged transaction gets its own explanation.

`TreeExplainer` is used specifically because it computes exact Shapley values for tree-based models in polynomial time, without the approximation overhead of kernel-based methods. This makes it fast enough for real-time inference in the Streamlit app.

### What It Produces
For every flagged transaction:

- **Risk factor bar chart** — top features ranked by absolute SHAP value, colored by direction (red = increases suspicion, teal = reduces it)
- **Contribution waterfall** — shows how each feature shifts the score step-by-step from the base value (average model output) to the final risk score
- **Risk gauge** — probability score visualized as a gauge mapped to Critical / High / Medium / Low

### Why This Matters
This is what makes the model deployable in a real AML workflow. A black-box score would be rejected by compliance teams — SHAP output gives analysts the "because" behind each alert, satisfies model interpretability requirements in regulated financial environments, and surfaces which features are doing the most work in the model overall.

---

## Multi-Agent Investigation Pipeline

This is what separates SynapseAML from a standalone classifier. Once the model flags a transaction, a **CrewAI multi-agent system** automates the entire post-prediction workflow — work that would otherwise require manual analyst time.

```
Transaction Input
      │
      ▼
┌──────────────────────────────────────┐
│   XGBoost Pipeline                   │
│   → Risk Score + SHAP Values         │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  CrewAI Sequential Agent Pipeline                    │
│                                                      │
│  1. Compliance Analyst                               │
│     └─ Screens entity name + country against         │
│        AML policy and sanctions rules                │
│                                                      │
│  2. Predictor Agent                                  │
│     └─ Invokes XGBoost model, returns risk score     │
│                                                      │
│  3. Analyst Agent                                    │
│     └─ Runs SHAP analysis + pattern detection        │
│        Generates LLM-backed investigation report     │
│        (Gemini 2.0 Flash)                            │
│                                                      │
│  4. Case Manager Agent                               │
│     └─ Creates structured case record with           │
│        priority, status, risk score, timestamp       │
└──────────────────────────────────────────────────────┘
      │
      ▼
 Markdown Investigation Report  +  Cases Dashboard
```

### Why Multi-Agent
A single monolithic script could technically run these steps sequentially. The multi-agent design was chosen because each concern — compliance screening, ML inference, explainability analysis, case management — is genuinely independent and benefits from isolated context. Each agent that interacts with the ML layer does so through programmatic tools that load `model.joblib` directly, bridging the gap between deterministic ML inference and LLM-based reasoning. CrewAI's `Process.sequential` orchestration keeps execution order deterministic, which matters for auditability in a compliance context.

---

## Suspicious Pattern Detection

On top of the model score, rule-based checks flag known AML typologies:

| Pattern | Signal | Severity |
| --- | --- | --- |
| Structuring | Amount $8,000–$9,999 (just below $10K reporting threshold) | High |
| Currency Layering | High-value transaction with cross-currency conversion | Medium |
| Unusual Routing | Bank pair matches known suspicious routes | High |
| Off-hours Activity | Transaction on weekends or outside 8 AM–6 PM | Low |

---

## Tech Stack

| Component | Technology |
| --- | --- |
| Core ML Model | XGBoost + scikit-learn Pipeline |
| Explainability | SHAP (`TreeExplainer`), Matplotlib, Plotly |
| Agent Framework | CrewAI |
| LLM (report generation) | Google Gemini 2.0 Flash |
| UI | Streamlit |
| Data Processing | Pandas, NumPy |

---

## Project Structure

```
synapseAML/
├── XGBoost.ipynb           # Model training, evaluation, and serialization
├── explainer.py            # SHAP analysis, pattern detection, report generation
├── app.py                  # Streamlit app — input form, results, case dashboard
├── cases_data.json         # Persistent case storage
├── requirements.txt
└── src/
    ├── agents.py           # CrewAI agent definitions
    ├── tasks.py            # CrewAI task definitions
    └── model.joblib        # Serialized XGBoost pipeline
```

---

## Getting Started

```bash
git clone https://github.com/sowmyaa88/synapseAML.git
cd synapseAML
pip install -r requirements.txt
```

Create a `.env` file:
```env
GEMINI_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

---

## Usage

1. Enter transaction details (bank codes, accounts, amounts, currencies, payment format, date/time)
2. Enter entity name and country for compliance screening
3. Click **Check Compliance** — the full agent pipeline runs sequentially
4. **Compliance Validation tab** — view risk level, status, and recommendation
5. **Report tab** — download the full SHAP + LLM investigation report
6. **Cases Dashboard** — monitor all flagged cases with priority breakdown and analytics
