from crewai import Task
from src.Tools.custom_tool import report_tool, predictor_tool, compliance_analyst_tool, case_manager_tool
import json
from pydantic import BaseModel

# Structured output for Compliance
class compliance(BaseModel):
    status: str
    risk_level: str
    details: str
    recommendation: str

# Structured output for Case Management
class case(BaseModel):
    case_id: str
    priority: str
    risk_score: int

def Compliance_Validation_Task(agent, entity_name: str, entity_country: str) -> Task:
    return Task(
        description=f"""
        Perform a complete regulatory compliance check on the transaction by validating 
        against both sanctions lists and PEP databases. Analyze the risk factors and 
        provide a comprehensive compliance recommendation based on all findings.
        
        Check entity name: {entity_name}
        Check entity country: {entity_country}
        
        IMPORTANT: Your output must be a valid JSON object.
        """,
        expected_output="""
        A valid JSON object with:
        - status (CLEARED, FLAGGED, or BLOCKED)
        - risk_level (low, medium, high)
        - details (explanation of any matches found)
        - recommendation (specific handling instructions)
        """,
        agent=agent,
        output_json=compliance,
        tools=[compliance_analyst_tool]
    )

def Prediction_task(agent, from_bank: int, account: str, to_bank: int, account_dest: str, 
                  amount_received: float, receiving_currency: str, amount_paid: float, 
                  payment_currency: str, payment_format: str, date: str, day: str, time: str) -> Task:
    return Task(
        description=f"""
        Analyze the transaction details provided and predict whether it's fraudulent/suspicious.
        
        Transaction details:
        - Sender bank code: {from_bank}
        - Sender account number: {account}
        - Receiver bank code: {to_bank}
        - Receiver account number: {account_dest}
        - Amount received: {amount_received}
        - Receiving currency: {receiving_currency}
        - Amount paid: {amount_paid}
        - Payment currency: {payment_currency}
        - Payment format: {payment_format}
        - Transaction date: {date}
        - Day of the week: {day}
        - Transaction time: {time}
        """,
        expected_output="""
        A prediction indicating whether the transaction is fraudulent or not, along with a 
        confidence score and the transaction details used for the prediction.
        """,
        agent=agent,
        tools=[predictor_tool]
    )

def Reporting_task(agent, from_bank: int, account: str, to_bank: int, account_dest: str, 
                 amount_received: float, receiving_currency: str, amount_paid: float, 
                 payment_currency: str, payment_format: str, date: str, day: str, time: str) -> Task:
    return Task(
        description=f"""
        Use these transaction details to generate a detailed forensic AML report.

        Transaction details:
        - Sender bank code: {from_bank}
        - Sender account number: {account}
        - Receiver bank code: {to_bank}
        - Receiver account number: {account_dest}
        - Amount received: {amount_received}
        - Receiving currency: {receiving_currency}
        - Amount paid: {amount_paid}
        - Payment currency: {payment_currency}
        - Payment format: {payment_format}
        - Transaction date: {date}
        - Day of the week: {day}
        - Transaction time: {time}
        """,
        expected_output="""
        A confirmation message containing the path to the saved markdown report, 
        confirming that the detailed SHAP and pattern analysis has been documented.
        """,
        agent=agent,
        tools=[report_tool]
    )

def Case_creation_task(agent, transaction_data: dict) -> Task:
    return Task(
        description=f"""
        Create a new AML investigation case based on transaction data. 
        Assign a case ID, set status to 'new', and determine priority.

        Transaction Data: {transaction_data}

        NOTE: Do not create a formal case if the transaction is predicted as normal, 
        but still provide the JSON output with a 'Low' priority.
        """,
        expected_output="""
        Confirmation of case creation with case ID and priority score. 
        Must be returned as a valid JSON object.
        """,
        agent=agent,
        output_json=case,
        tools=[case_manager_tool]
    )