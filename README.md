# Automating AML Compliance with Agentic AI

> An AI-powered solution to automate Anti-Money Laundering (AML) workflows using modular, specialized agents built on CrewAI.

## 🏆 Award

**Winner of the 1st Place Prize** at the *Agentic AI Hackathon* conducted by **Techvantage.ai** in collaboration with **CrewAI**.

## 🚀 Project Overview

Manual AML compliance processes often result in delays, human errors, and overlooked red flags. This project offers a scalable and explainable multi-agent AI system to monitor, detect, and manage suspicious financial transactions efficiently.

Using CrewAI for orchestration, each agent is tasked with a specific AML responsibility — from prediction to validation and reporting — enabling a seamless and autonomous compliance pipeline.

## 🎯 Core Features

- **Agentic Workflow Orchestration:** CrewAI coordinates task-specific agents for streamlined AML operations.
- **Fraud Detection:** Uses XGBoost-based models to identify potentially fraudulent transactions.
- **Compliance Validation:** Validates entities against global sanctions and PEP (Politically Exposed Persons) databases.
- **Explainable Reporting:** GPT-4o and SHAP explain model decisions with rich markdown summaries.
- **Case Management:** Assigns risk-based priorities and tracks flagged cases.
- **Frontend Interface:** A user-friendly Streamlit dashboard for seamless interaction.

## 🧰 Tech Stack

- **AI & Agents:** CrewAI, GPT-4o, Pydantic
- **Machine Learning:** XGBoost, SHAP, Scikit-learn
- **Visualization:** Streamlit, Plotly, Seaborn, Matplotlib
- **Data Handling:** Pandas, NumPy, Joblib, JSON, Markdown
- **Environment:** Python 3.11, dotenv

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git https://github.com/VenkataSakethDakuri/AML_Crew
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

## 🎥 Demo

Watch the full walkthrough on Loom: [Demo Video](https://www.loom.com/share/bc340f2d582b410a9e93bd7c7cadad9a?sid=b335a460-f81f-4243-b55a-219faccc20bc)

## 📈 Performance

- **Precision (Fraudulent):** 0.82  
- **Recall (Fraudulent):** 0.93  
- **F1-Score (Fraudulent):** 0.87  
> Model effectively detects laundering with high recall while reducing false positives.

## 🧠 Future Enhancements

- Graph-based anomaly detection  
- Real-time sanctions/PEP data updates  
- Custom risk modeling for institutional compliance

## 📄 License

MIT License. See `LICENSE` for details.

---

*Crafted for the Techvantage.ai BFSI Hackathon using CrewAI.* ✨
