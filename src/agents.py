import os
from crewai import Agent, LLM
from src.Tools.custom_tool import predictor_tool, compliance_analyst_tool, case_manager_tool, report_tool
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key= os.getenv("GEMINI_API_KEY"), # Your key directly here
    base_url="https://generativelanguage.googleapis.com/v1beta"
)

Predictor = Agent(
    role="Transaction Predictor",
    goal="To check if the transaction is suspicious or not.",
    backstory="You are a data scientist with a strong background in machine learning and predictive modeling. You have experience working with large datasets and building models that can accurately predict outcomes based on historical data. Your goal is to identify suspicious transactions using the predictor tool.",
    llm=gemini_llm,
    tools=[predictor_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False
)

Analyst = Agent(
    role="Data Analyst",
    goal="Creating clear and concise reports using feature importance.",
    backstory="You are a data analyst experienced in generating reports that provide valuable insights. Your goal is to analyze transaction results and use the report tool to create detailed documentation for the AML team.",
    llm=gemini_llm,
    tools=[report_tool],
    verbose=True,
    allow_delegation=False
)

CaseManager = Agent(
    role="AML Case Manager",
    goal="Efficiently manage, track, and prioritize AML investigation cases.",
    backstory="You are an experienced AML compliance officer. You excel at organizing complex investigations and using the case manager tool to prioritize high-risk cases based on model confidence.",
    llm=gemini_llm,
    tools=[case_manager_tool],
    verbose=True,
    allow_delegation=False
)

ComplianceAnalyst = Agent(
    role="Compliance Analyst",
    goal="Validating entities against global sanction lists and PEP databases.",
    backstory="You are a compliance analyst with a strong background in regulatory compliance. You use the compliance analyst tool to check entities against sanction lists and ensure everything is legal.",
    llm=gemini_llm,
    tools=[compliance_analyst_tool],
    max_iter=3,
    verbose=True,
    allow_delegation=False
)