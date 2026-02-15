from crewai.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import datetime
import joblib
import os
import ast
import json

@tool("My Custom Tool")
def my_custom_tool(argument: str) -> str:
    """Clear description for what this tool is useful for, your agent will need this information to use it.
    
    Args:
        argument (str): Description of the argument.
    
    Returns:
        str: Tool output
    """
    return "this is an example of a tool output, ignore it and move along."

@tool("Predictor Tool")
def predictor_tool(from_bank: int, account: str, to_bank: int, account_dest: str, 
                   amount_received: float, receiving_currency: str, amount_paid: float, 
                   payment_currency: str, payment_format: str, date: str, day: str, time: str) -> dict:
    """This tool predicts whether a transaction is fraudulent or not based on transaction details.
    
    Args:
        from_bank (int): Sender bank code
        account (str): Sender account number
        to_bank (int): Receiver bank code
        account_dest (str): Receiver account number
        amount_received (float): Amount received in transaction
        receiving_currency (str): Currency of the received amount
        amount_paid (float): Amount paid in transaction
        payment_currency (str): Currency of the payment
        payment_format (str): Format of payment (e.g., Cheque, Wire)
        date (str): Transaction date (YYYY-MM-DD)
        day (str): Day of the week
        time (str): Transaction time (HH:MM:SS)
    
    Returns:
        dict: Prediction result including prediction and confidence score
    """
    transaction_data = {
        'From Bank': [from_bank],
        'Account': [account],
        'To Bank': [to_bank],
        'Account.1': [account_dest],
        'Amount Received': [amount_received],
        'Receiving Currency': [receiving_currency],
        'Amount Paid': [amount_paid],
        'Payment Currency': [payment_currency],
        'Payment Format': [payment_format],
        'Date': [date],
        'Day': [day],
        'Time': [time]
    }
    transaction_df = pd.DataFrame(transaction_data)
    model = joblib.load("src/model.joblib")
    prediction = model.predict(transaction_df)
    prediction_proba = model.predict_proba(transaction_df)
    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    confidence_percentage = round(confidence * 100, 2)

    return {
        'prediction': int(prediction[0]),
        'confidence_percentage': confidence_percentage,
        'transaction_data': transaction_data
    }

@tool("Report Tool")
def report_tool(from_bank: int, account: str, to_bank: int, account_dest: str, 
                amount_received: float, receiving_currency: str, amount_paid: float, 
                payment_currency: str, payment_format: str, date: str, day: str, time: str) -> str:
    """This tool analyzes a transaction and creates a detailed report based on the transaction details.
    
    Args:
        from_bank (int): Sender bank code
        account (str): Sender account number
        to_bank (int): Receiver bank code
        account_dest (str): Receiver account number
        amount_received (float): Amount received in transaction
        receiving_currency (str): Currency of the received amount
        amount_paid (float): Amount paid in transaction
        payment_currency (str): Currency of the payment
        payment_format (str): Format of payment (e.g., Cheque, Wire)
        date (str): Transaction date (YYYY-MM-DD)
        day (str): Day of the week
        time (str): Transaction time (HH:MM:SS)
    
    Returns:
        str: Status message with report path
    """
    try:
        transaction_data = {
            'From Bank': [from_bank],
            'Account': [account],
            'To Bank': [to_bank],
            'Account.1': [account_dest],
            'Amount Received': [amount_received],
            'Receiving Currency': [receiving_currency],
            'Amount Paid': [amount_paid],
            'Payment Currency': [payment_currency],
            'Payment Format': [payment_format],
            'Date': [date],
            'Day': [day],
            'Time': [time]
        }
        
        # Create DataFrame for analysis
        sample_df = pd.DataFrame(transaction_data)
        
        # Set output directory
        output_dir = './report'
        os.makedirs(output_dir, exist_ok=True)
        
        # Import and use the analyze function
        from explainer import analyze_transaction_and_create_report
        report_path = analyze_transaction_and_create_report(sample_df, output_dir=output_dir)
        
        return f"Report created successfully at {report_path}."
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        return f"Error creating report: {str(e)}"

    
    

# Global variables for case management
_cases = {}
_case_counter = 1

@tool("Case Manager Tool")
def case_manager_tool(transaction_data: dict, prediction_confidence: float, prediction_result: int) -> str:
    """This tool creates AML investigation cases with priority based on prediction confidence.
    
    Args:
        transaction_data (dict): Transaction data for creating a new case
        prediction_confidence (float): Confidence score from prediction to determine priority
        prediction_result (int): Prediction result (0 for normal, 1 for suspicious)
    
    Returns:
        str: Case creation status message
    """
    global _cases, _case_counter
    
    if prediction_result == 0:
        return "No case created - transaction predicted as normal"
    
    case_id = f"AML-{_case_counter}"
    _case_counter += 1
    
    # NEW LOGIC: Ensure priority aligns with the risk result
    if prediction_result == 1:
        if prediction_confidence >= 90:
            priority = "Critical"
        elif prediction_confidence >= 70:
            priority = "High"  # This will correctly catch your 81% score
        else:
            priority = "Medium" # Suspicious transactions now have a 'Medium' floor
    else:
        priority = "Low" # Only non-suspicious outcomes can be Low
    
    case = {
        "case_id": case_id,
        "status": "new",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_updated": datetime.datetime.now().isoformat(),
        "transaction_data": transaction_data,
        "priority": priority,
        "prediction_confidence": prediction_confidence,
        "history": [
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "case_created",
                "details": f"New case created with {priority} priority based on {prediction_confidence}% prediction confidence"
            }
        ]
    }
    
    _cases[case_id] = case
    return {
        "case_id": case_id,
        "priority": priority,
        "prediction_confidence": prediction_confidence
    }

# Global variables for compliance analysis
_sanctions_db = []
_pep_db = []

@tool("Compliance Analyst Tool")
def compliance_analyst_tool(entity_name: str, entity_country: str) -> str:
    """This tool validates entities against global sanction lists and PEP databases.
    
    Args:
        entity_name (str): Name of individual or entity to check
        entity_country (str): Country of the individual or entity
    
    Returns:
        str: Compliance check results
    """
    global _sanctions_db, _pep_db
    
    if not entity_name or not entity_country:
        return "Error: Missing entity_name or entity_country. Please provide valid values."
    
    results = {}
    sanctions_results = _check_sanctions(entity_name, entity_country)
    pep_results = _check_pep(entity_name, entity_country)
    
    results['sanctions_check'] = sanctions_results
    results['pep_check'] = pep_results
    
    if sanctions_results['match_found']:
        results['status'] = 'BLOCKED - SANCTIONS MATCH'
        results['risk_level'] = 'high'
    elif pep_results['match_found']:
        results['status'] = 'FLAGGED - PEP MATCH' 
        results['risk_level'] = 'medium'
    else:
        results['status'] = 'CLEARED'
        results['risk_level'] = 'low'
    
    results['recommendation'] = _generate_recommendation(results)
    
    return str(results)

def _check_sanctions(entity_name: str, entity_country: str) -> dict:
    global _sanctions_db
    match = any(
        entity_name.lower() in entity['name'].lower() and 
        entity_country.lower() == entity['country'].lower()
        for entity in _sanctions_db
    )
    
    if match:
        matched_entities = [
            entity for entity in _sanctions_db 
            if entity_name.lower() in entity['name'].lower() and
            entity_country.lower() == entity['country'].lower()
        ]
        return {
            'match_found': True,
            'matched_entities': matched_entities,
            'confidence': 'high'
        }
    
    return {
        'match_found': False,
        'matched_entities': [],
        'confidence': None
    }

def _check_pep(entity_name: str, entity_country: str) -> dict:
    global _pep_db
    match = any(
        entity_name.lower() in entity['name'].lower() and
        entity_country.lower() == entity['country'].lower()
        for entity in _pep_db
    )
    
    if match:
        matched_entities = [
            entity for entity in _pep_db
            if entity_name.lower() in entity['name'].lower() and
            entity_country.lower() == entity['country'].lower()
        ]
        return {
            'match_found': True,
            'matched_entities': matched_entities,
            'confidence': 'high'
        }
    
    return {
        'match_found': False,
        'matched_entities': [],
        'confidence': None
    }

def _generate_recommendation(results: dict) -> str:
    if results['status'] == 'BLOCKED - SANCTIONS MATCH':
        return "Block transaction and file SAR immediately"
    elif results['status'] == 'FLAGGED - PEP MATCH':
        return "Enhanced due diligence required before proceeding"
    return "Transaction cleared - no further action needed"

def _initialize_sanctions_db():
    global _sanctions_db
    _sanctions_db = [
        {
            'name': 'Restricted Trading Company',
            'country': 'Country A',
            'list': 'OFAC SDN',
            'reason': 'Proliferation financing'
        },
        {
            'name': 'Global Sanctioned Bank',
            'country': 'Country B', 
            'list': 'EU Consolidated',
            'reason': 'Supporting terrorism'
        }
    ]

def _initialize_pep_db():
    global _pep_db
    _pep_db = [
        {
            'name': 'Minister Finance',
            'country': 'Country E',
            'position': 'Finance Minister',
            'risk_level': 'high'
        },
        {
            'name': 'Governor Central',
            'country': 'Country F',
            'position': 'Central Bank Governor', 
            'risk_level': 'medium'
        }
    ]

# Initialize databases
_initialize_sanctions_db()
_initialize_pep_db()
