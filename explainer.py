import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
from crewai import LLM

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load the saved model
model = joblib.load("src/model.joblib")

# Human-readable feature name mapping
FEATURE_DISPLAY_NAMES = {
    # Map technical feature names to business-friendly names
    # Basic features
    'From Bank': 'Sender Bank',
    'To Bank': 'Receiver Bank', 
    'Account': 'Account Number',
    'Amount': 'Transaction Amount',
    'Receiving Currency': 'Destination Currency',
    'Payment Currency': 'Source Currency',
    'Payment Format': 'Transfer Method',
    'Day': 'Day of Week',
    'Date': 'Transaction Date',
    'Time': 'Transaction Time',
    
    # Example one-hot encoded feature mappings - update these based on your actual model features
    'One_Hot_Encoding__From Bank_Bank A': 'Sender: Bank A',
    'One_Hot_Encoding__From Bank_Bank B': 'Sender: Bank B',
    'One_Hot_Encoding__To Bank_Bank Z': 'Receiver: Bank Z',
    'One_Hot_Encoding__Payment Format_Wire': 'Wire Transfer',
    'One_Hot_Encoding__Payment Format_Cash': 'Cash Payment',
    'One_Hot_Encoding__Day_Sunday': 'Sunday Transaction',
    'One_Hot_Encoding__Day_Saturday': 'Saturday Transaction',
    'One_Hot_Encoding__Receiving Currency_USD': 'To USD',
    'One_Hot_Encoding__Payment Currency_EUR': 'From EUR'
}

def get_readable_feature_name(feature):
    """Convert technical feature names to business-friendly terms"""
    # Check direct matches first
    if feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]
    
    # Handle one-hot encoded features with pattern matching
    if feature.startswith('One_Hot_Encoding__'):
        # Extract the category and value from the feature name
        parts = feature.split('_')
        if len(parts) >= 4:
            category = parts[2]
            value = parts[3]
            
            # Create readable name based on category
            if category == 'From Bank':
                return f"Sender: {value}"
            elif category == 'To Bank':
                return f"Receiver: {value}"
            elif category == 'Payment Format':
                return f"Method: {value}"
            elif category == 'Day':
                return f"{value} Transaction"
            elif category == 'Receiving Currency':
                return f"To {value}"
            elif category == 'Payment Currency':
                return f"From {value}"
    
    # Default formatting for unknown features
    return feature.replace('_', ' ').replace('One Hot Encoding ', '').title()

def detect_suspicious_patterns(transaction, explanations=None):
    """
    Identify potential money laundering patterns based on transaction data
    """
    patterns = []
    tx = transaction.iloc[0]
    
    # Pattern 1: Large amount with currency conversion
    if hasattr(tx, 'Amount') and tx.get('Amount', 0) > 10000:
        if tx.get('Payment Currency') != tx.get('Receiving Currency'):
            patterns.append({
                'pattern': 'Currency Conversion',
                'description': 'High-value transaction with currency conversion',
                'severity': 'Medium',
                'details': f"Transaction amount (${tx.get('Amount'):,.2f}) exceeds $10,000 threshold and involves currency conversion from {tx.get('Payment Currency')} to {tx.get('Receiving Currency')}"
            })
    
    # Pattern 2: Unusual bank routing
    unusual_bank_pairs = [('Bank X', 'Bank Y'), ('Bank A', 'Bank Z')]  # Example pairs
    from_bank = tx.get('From Bank')
    to_bank = tx.get('To Bank')
    
    if (from_bank, to_bank) in unusual_bank_pairs:
        patterns.append({
            'pattern': 'Unusual Routing',
            'description': f'Suspicious bank routing pattern detected',
            'severity': 'High',
            'details': f"Transaction route between {from_bank} and {to_bank} matches known suspicious patterns often used for layering"
        })
    
    # Pattern 3: Structured transaction (just below reporting threshold)
    if hasattr(tx, 'Amount') and 8000 <= tx.get('Amount', 0) <= 9999:
        patterns.append({
            'pattern': 'Structured Transaction',
            'description': 'Amount just below mandatory reporting threshold',
            'severity': 'High',
            'details': f"Transaction amount (${tx.get('Amount'):,.2f}) appears structured to avoid the $10,000 reporting threshold"
        })
    
    # Pattern 4: Weekend transactions
    if hasattr(tx, 'Day') and tx.get('Day') in ['Saturday', 'Sunday']:
        patterns.append({
            'pattern': 'Off-hours Activity',
            'description': 'Transaction conducted outside business days',
            'severity': 'Low',
            'details': f"Transaction occurred on {tx.get('Day')}, which is unusual for legitimate business operations"
        })
    
    # Pattern 5: After-hours transaction
    if hasattr(tx, 'Time') and hasattr(tx.get('Time'), 'hour'):
        hour = tx.get('Time').hour
        if hour < 8 or hour > 18:
            patterns.append({
                'pattern': 'After-hours Activity',
                'description': 'Transaction outside normal business hours',
                'severity': 'Medium',
                'details': f"Transaction time ({tx.get('Time')}) falls outside standard business hours (8 AM - 6 PM)"
            })
    
    return patterns

def identify_risk_factors(transaction):
    """
    Identify risk factors in the transaction
    """
    risk_factors = []
    tx = transaction.iloc[0]
    
    # Geographic risk
    high_risk_countries = ['Country X', 'Country Y']  # Example high-risk countries
    if hasattr(tx, 'country') and any(country in str(tx.get('country', '')) for country in high_risk_countries):
        risk_factors.append({
            'factor': 'High-risk Jurisdiction',
            'description': 'Transaction involves high-risk country',
            'severity': 'High',
            'details': f"The transaction involves a jurisdiction ({tx.get('country')}) with elevated money laundering risk"
        })
    
    # Payment method risk
    if hasattr(tx, 'Payment Format') and hasattr(tx, 'Amount'):
        if tx.get('Payment Format') == 'Cash' and tx.get('Amount', 0) > 5000:
            risk_factors.append({
                'factor': 'Cash Transaction',
                'description': 'Large cash payment',
                'severity': 'Medium',
                'details': f"Cash payment of ${tx.get('Amount'):,.2f} exceeds the typical threshold for cash transactions"
            })
    
    # Time-based risk
    if hasattr(tx, 'Time') and hasattr(tx.get('Time'), 'hour'):
        hour = tx.get('Time').hour
        if hour < 9 or hour > 17:
            risk_factors.append({
                'factor': 'Unusual Timing',
                'description': 'Transaction outside business hours',
                'severity': 'Low',
                'details': f"Transaction conducted at {tx.get('Time')} is outside regular business hours"
            })
    
    # Currency conversion risk
    if hasattr(tx, 'Payment Currency') and hasattr(tx, 'Receiving Currency'):
        if tx.get('Payment Currency') != tx.get('Receiving Currency'):
            risk_factors.append({
                'factor': 'Currency Conversion',
                'description': 'Multi-currency transaction',
                'severity': 'Low',
                'details': f"Currency conversion from {tx.get('Payment Currency')} to {tx.get('Receiving Currency')} may indicate layering activity"
            })
    
    # Weekend activity
    if hasattr(tx, 'Day') and tx.get('Day') in ['Saturday', 'Sunday']:
        risk_factors.append({
            'factor': 'Weekend Activity',
            'description': 'Non-business day transaction',
            'severity': 'Low',
            'details': f"Transaction conducted on {tx.get('Day')}, unusual for standard business operations"
        })
    
    return risk_factors

def analyze_suspicious_transaction(transaction, model, output_dir='./analysis_output'):
    """
    Extract detailed insights about why a transaction is marked as suspicious.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make prediction
    prediction = model.predict(transaction)
    probabilities = model.predict_proba(transaction)
    suspicious_prob = probabilities[0][1]
    
    # Check if transaction is flagged as suspicious
    if prediction[0] == 0:
        return {
            "is_suspicious": False,
            "probability": float(suspicious_prob),
            "message": "Transaction classified as NORMAL",
            "transaction_details": transaction.iloc[0].to_dict()
        }
    
    # Store transaction details and prediction info
    result = {
        "is_suspicious": True,
        "probability": float(suspicious_prob),
        "transaction_details": transaction.iloc[0].to_dict(),
        "explanations": {},
        "visualization_paths": {}
    }
    
    # Extract components from pipeline
    transformer = model.named_steps['Transformer']
    xgb_model = model.named_steps['Estimator']
    
    # Use SHAP values to explain the prediction
    try:
        # Transform the data
        transformed_data = transformer.transform(transaction)
        if hasattr(transformed_data, 'toarray'):
            transformed_data = transformed_data.toarray()
        
        # Get feature names after transformation
        try:
            raw_feature_names = transformer.get_feature_names_out()
        except:
            raw_feature_names = [f"feature_{i}" for i in range(transformed_data.shape[1])]
        
        # Convert to readable feature names
        feature_names = [get_readable_feature_name(f) for f in raw_feature_names]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(transformed_data)
        
        # Handle binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use values for class 1 (suspicious)
            shap_values_class1 = shap_values[1]
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            shap_values_class1 = shap_values
            expected_value = explainer.expected_value
        
        # Create DataFrame with SHAP values and readable names
        shap_df = pd.DataFrame({
            'feature': raw_feature_names,
            'display_name': feature_names,
            'shap_value': shap_values_class1[0],
            'impact': np.abs(shap_values_class1[0])
        }).sort_values('impact', ascending=False)
        
        # Add direction (increases or decreases suspicion)
        shap_df['direction'] = shap_df['shap_value'].apply(
            lambda x: "Increases Risk" if x > 0 else "Reduces Risk"
        )
        
        # Save top contributors to result
        result['explanations']['shap_analysis'] = {
            'base_value': float(expected_value),
            'contributors': shap_df[['display_name', 'shap_value', 'impact', 'direction']].head(15).to_dict('records')
        }
        
        # Generate visualizations with better labels
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        plt.title('Key Transaction Risk Factors', fontsize=16, pad=20)
        
        # Create custom summary plot to use display names
        shap_importance = shap_df.head(10).copy()
        # Sort by absolute impact for visualization
        shap_importance = shap_importance.sort_values('impact')
        
        colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in shap_importance['shap_value']]
        plt.barh(shap_importance['display_name'], shap_importance['impact'], color=colors)
        plt.xlabel('Impact on Risk Assessment', fontsize=12)
        plt.ylabel('Transaction Attributes', fontsize=12)
        
        # Add 'Increases/Reduces Risk' annotations
        for i, row in enumerate(shap_importance.itertuples()):
            direction = "Increases Risk" if row.shap_value > 0 else "Reduces Risk"
            plt.text(row.impact + 0.005, i, direction, va='center', fontsize=9)
        
        shap_plot_path = os.path.join(output_dir, 'risk_factors.png')
        plt.savefig(shap_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        result['visualization_paths']['risk_factors'] = shap_plot_path
        
        # 2. Risk Contribution Waterfall
        plt.figure(figsize=(12, 9))
        
        # Create custom waterfall chart
        sorted_impacts = shap_df.head(10).copy()
        baseline = expected_value
        cumulative = baseline
        
        # Set up plot
        plt.barh([sorted_impacts['display_name'].iloc[0]], [baseline], color='#888888', alpha=0.6)
        plt.text(baseline, 0, f'Base Risk\n{baseline:.3f}', ha='center', va='center', fontsize=9)
        
        # Plot each contribution
        y_pos = np.arange(1, len(sorted_impacts) + 1)
        for i, row in enumerate(sorted_impacts.itertuples(), 1):
            color = '#ff6b6b' if row.shap_value > 0 else '#4ecdc4'
            plt.barh([row.display_name], [abs(row.shap_value)], left=cumulative if row.shap_value > 0 else cumulative - row.shap_value, 
                    color=color)
            
            new_value = cumulative + row.shap_value
            plt.text(cumulative + row.shap_value/2 if row.shap_value > 0 else cumulative - row.shap_value/2, 
                    i-1, f'{row.shap_value:+.3f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(row.shap_value) > 0.15 else 'black')
            
            cumulative = new_value
        
        # Final value
        plt.barh(['Final Risk Score'], [0.1], left=cumulative-0.05, color='#FF9F1C')
        plt.text(cumulative, len(sorted_impacts), f'Final: {cumulative:.3f}', ha='center', va='center', fontsize=9)
        
        plt.title('How Each Factor Contributes to Risk Assessment', fontsize=16)
        plt.xlabel('Cumulative Risk Score', fontsize=12)
        plt.tight_layout()
        waterfall_path = os.path.join(output_dir, 'risk_breakdown.png')
        plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
        plt.close()
        result['visualization_paths']['risk_breakdown'] = waterfall_path
        
        # 3. Risk Score Gauge
        plt.figure(figsize=(10, 6))
        
        # Create a gauge-like visualization using a horizontal bar
        risk_score = suspicious_prob * 100
        
        # Define risk categories
        categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        thresholds = [0, 25, 50, 75, 100]
        colors = ['#4ecdc4', '#FFD166', '#ff6b6b', '#9C0D38']
        
        # Plot the background bars for each category
        for i in range(len(categories)):
            plt.barh([0], [thresholds[i+1] - thresholds[i]], left=thresholds[i], 
                    color=colors[i], alpha=0.6, height=0.6)
            # Add category labels
            plt.text((thresholds[i] + thresholds[i+1])/2, 0.8, categories[i], 
                    ha='center', va='center', fontsize=11)
        
        # Add the pointer (triangle) to show where this transaction falls
        pointer_height = 0
        plt.scatter(risk_score, pointer_height, marker='v', s=400, color='black')
        
        # Add the risk score
        plt.text(risk_score, -0.4, f'{risk_score:.1f}%', ha='center', va='center', 
                fontweight='bold', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        plt.title('Transaction Risk Level', fontsize=16)
        plt.xlim(0, 100)
        plt.ylim(-0.8, 1.2)
        plt.axis('off')
        
        risk_gauge_path = os.path.join(output_dir, 'risk_gauge.png')
        plt.savefig(risk_gauge_path, bbox_inches='tight', dpi=150)
        plt.close()
        result['visualization_paths']['risk_gauge'] = risk_gauge_path
        
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        # Fallback to XGBoost feature importance
        try:
            # Get feature names
            try:
                raw_feature_names = transformer.get_feature_names_out()
                feature_names = [get_readable_feature_name(f) for f in raw_feature_names]
            except:
                feature_names = [f"Feature {i}" for i in range(len(xgb_model.feature_importances_))]
            
            # Create feature importance DataFrame with readable names
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Add to results
            result['explanations']['feature_importance'] = importance_df.head(15).to_dict('records')
            result['explanations']['error_message'] = f"Could not compute detailed SHAP values: {str(e)}"
            
            # Generate feature importance plot with better labels
            plt.figure(figsize=(12, 8))
            top_importance = importance_df.head(10).copy()
            
            # Sort for better visualization
            top_importance = top_importance.sort_values('importance')
            plt.barh(top_importance['feature'], top_importance['importance'], color='#4361ee')
            
            plt.title('Most Important Risk Indicators', fontsize=16)
            plt.xlabel('Relative Importance', fontsize=12)
            plt.ylabel('Transaction Attributes', fontsize=12)
            
            # Add value labels
            for i, value in enumerate(top_importance['importance']):
                plt.text(value + 0.01, i, f'{value:.3f}', va='center')
            
            importance_plot_path = os.path.join(output_dir, 'risk_indicators.png')
            plt.savefig(importance_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            result['visualization_paths']['risk_indicators'] = importance_plot_path
            
        except Exception as xgb_error:
            print(f"Error in XGBoost importance analysis: {str(xgb_error)}")
            result['explanations']['error_message'] = f"Failed to generate explanations: {str(e)}. XGBoost analysis error: {str(xgb_error)}"
    
    # Add pattern detection
    try:
        patterns = detect_suspicious_patterns(transaction, result['explanations'])
        if patterns:
            result['suspicious_patterns'] = patterns
            
            # Create pattern visualization if patterns were detected
            if len(patterns) > 0:
                plt.figure(figsize=(12, 8))
                
                # Prepare data for the plot
                pattern_names = [p['pattern'] for p in patterns]
                severity_values = [3 if p['severity'] == 'High' else 2 if p['severity'] == 'Medium' else 1 for p in patterns]
                severity_colors = ['#ff6b6b' if p['severity'] == 'High' else '#FFD166' if p['severity'] == 'Medium' else '#4ecdc4' for p in patterns]
                
                # Create horizontal bar chart
                plt.barh(pattern_names, severity_values, color=severity_colors)
                plt.xlabel('Risk Severity (1=Low, 2=Medium, 3=High)', fontsize=12)
                plt.title('Detected Suspicious Patterns', fontsize=16)
                plt.xlim(0, 4)
                
                # Replace x-ticks with severity labels
                plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'])
                
                # Add descriptions as annotations
                for i, pattern in enumerate(patterns):
                    plt.text(3.1, i, pattern['description'], va='center', fontsize=10)
                
                pattern_plot_path = os.path.join(output_dir, 'suspicious_patterns.png')
                plt.savefig(pattern_plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                result['visualization_paths']['suspicious_patterns'] = pattern_plot_path
    except Exception as pattern_error:
        print(f"Error in pattern detection: {str(pattern_error)}")
        result['pattern_detection_error'] = str(pattern_error)
    
    # Add risk factors and anomaly detection
    try:
        risk_factors = identify_risk_factors(transaction)
        if risk_factors:
            result['risk_factors'] = risk_factors
            
            # Create risk factor visualization
            if len(risk_factors) > 0:
                # Create pie chart of risk severities
                severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
                for risk in risk_factors:
                    severity_counts[risk['severity']] += 1
                
                # Only create chart if we have risk factors
                if sum(severity_counts.values()) > 0:
                    plt.figure(figsize=(10, 7))
                    
                    # Prepare data for pie chart
                    labels = [f"{k} ({v})" for k, v in severity_counts.items() if v > 0]
                    sizes = [v for v in severity_counts.values() if v > 0]
                    colors = ['#ff6b6b', '#FFD166', '#4ecdc4']
                    
                    # Create pie chart
                    patches, texts, autotexts = plt.pie(
                        sizes, 
                        labels=labels,
                        colors=colors,
                        autopct='%1.1f%%',
                        startangle=90,
                        shadow=False,
                    )
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.axis('equal')
                    plt.title('Risk Factor Severity Distribution', fontsize=16)
                    
                    # Make text more readable
                    for text in texts:
                        text.set_fontsize(12)
                    for autotext in autotexts:
                        autotext.set_fontsize(12)
                        autotext.set_fontweight('bold')
                    
                    risk_pie_path = os.path.join(output_dir, 'risk_severity.png')
                    plt.savefig(risk_pie_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    result['visualization_paths']['risk_severity'] = risk_pie_path
    except Exception as risk_error:
        print(f"Error in risk factor analysis: {str(risk_error)}")
        result['risk_detection_error'] = str(risk_error)
        
    return result

def create_llm_prompt(analysis_result):
    """
    Create a detailed prompt for GPT-4o mini based on the transaction analysis
    """
    if not analysis_result['is_suspicious']:
        return f"""
        You are a financial crime analyst specializing in anti-money laundering (AML). 
        You've been asked to analyze a transaction that was evaluated by our ML model.
        
        The transaction was classified as NORMAL with a probability of {analysis_result['probability']*100:.2f}% of being suspicious.
        
        Transaction details:
        {json.dumps(analysis_result['transaction_details'], indent=2, default=str)}
        
        Please explain briefly why this transaction appears to be normal and does not require further investigation.
        Format your response in markdown.
        """
    
    # For suspicious transactions, prepare a detailed prompt
    tx = analysis_result['transaction_details']
    
    # Format feature explanations
    feature_explanations = ""
    if 'shap_analysis' in analysis_result.get('explanations', {}):
        contributors = analysis_result['explanations']['shap_analysis']['contributors']
        
        feature_explanations += "Key Risk Factors (SHAP Analysis):\n"
        for i, contrib in enumerate(contributors[:10], 1):
            direction = "increases" if contrib['shap_value'] > 0 else "decreases"
            feature_explanations += f"- {contrib['display_name']}: {direction} risk (impact: {contrib['impact']:.4f})\n"
    
    # Format suspicious patterns
    pattern_text = ""
    if 'suspicious_patterns' in analysis_result and analysis_result['suspicious_patterns']:
        pattern_text += "Suspicious Patterns Detected:\n"
        for pattern in analysis_result['suspicious_patterns']:
            pattern_text += f"- {pattern['pattern']} ({pattern['severity']} risk): {pattern['description']}\n"
            if 'details' in pattern:
                pattern_text += f"  Details: {pattern['details']}\n"
    
    # Format risk factors
    risk_text = ""
    if 'risk_factors' in analysis_result and analysis_result['risk_factors']:
        risk_text += "Risk Factors Identified:\n"
        for risk in analysis_result['risk_factors']:
            risk_text += f"- {risk['factor']} ({risk['severity']} risk): {risk['description']}\n"
            if 'details' in risk:
                risk_text += f"  Details: {risk['details']}\n"
    
    # Create the prompt for the LLM
    prompt = f"""
You are a financial crime analyst specializing in anti-money laundering (AML) investigations.
You need to write a detailed report explaining why a transaction was flagged as suspicious.

## Transaction Details
The transaction was classified as SUSPICIOUS with {analysis_result['probability']*100:.2f}% confidence.

Transaction data:
{json.dumps(tx, indent=2, default=str)}

## Model Analysis
{feature_explanations}

{pattern_text}

{risk_text}

## Your Task
Based on the above information, write a comprehensive markdown report explaining why this transaction was flagged as suspicious. Your report should include:

1. An executive summary of the suspicious transaction
2. Detailed analysis of the key factors that contributed to the suspicion
3. Explanation of potential money laundering typologies or schemes that this transaction might indicate
4. Risk assessment (high/medium/low)
5. Recommended next steps for investigation
6. Additional data that might be useful to collect

Your report should be well-structured with markdown headings, bullet points, and emphasis where appropriate. Write in a professional tone suitable for compliance and investigation teams.
"""
    return prompt

def get_gemini_explanation(prompt):
    """Get explanation from Gemini via CrewAI"""
    try:
        # We already imported LLM at the top, so we just use it here
        llm = LLM(model="gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
        return llm.call([{"role": "user", "content": prompt}])
    except Exception as e:
        return f"**Error generating explanation**: {str(e)}"

def create_markdown_report(analysis_result, llm_explanation):
    """
    Create a comprehensive markdown report with transaction analysis and LLM explanation
    """
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the markdown content
    markdown_content = f"""# Transaction Risk Analysis Report
**Generated**: {timestamp}

## Transaction Summary
"""

    # Format transaction details in a table
    tx_details = analysis_result['transaction_details']
    markdown_content += "| Attribute | Value |\n|-----------|-------|\n"
    for key, value in tx_details.items():
        markdown_content += f"| {key.replace('_', ' ').title()} | {value} |\n"
    
    # Add prediction information
    prediction_prob = analysis_result['probability'] * 100
    markdown_content += f"\n**Risk Assessment**: "
    
    if analysis_result['is_suspicious']:
        markdown_content += f"⚠️ **SUSPICIOUS** with {prediction_prob:.1f}% confidence\n\n"
        
        # # Add risk gauge if available
        # if 'risk_gauge' in analysis_result.get('visualization_paths', {}):
        #     # Use just the basename of the image for better portability
        #     img_basename = os.path.basename(analysis_result['visualization_paths']['risk_gauge'])
        #     markdown_content += f"![Risk Level]({img_basename})\n\n"
    else:
        markdown_content += f"✅ **NORMAL** with {100-prediction_prob:.1f}% confidence\n\n"
    
    # If not suspicious, include brief explanation and exit
    if not analysis_result['is_suspicious']:
        markdown_content += "## Analysis\n\n"
        markdown_content += llm_explanation
        
        # Add timestamp
        markdown_content += f"\n\n---\n*Report generated on {timestamp}*"
        return markdown_content
    
    # For suspicious transactions, add more detailed analysis
    
    # Add risk factor visualizations
    markdown_content += "## Risk Analysis\n\n"
    
   # if 'risk_factors' in analysis_result['visualization_paths']:
        # Use just the basename of the image
        #img_basename = os.path.basename(analysis_result['visualization_paths']['risk_factors'])
        #markdown_content += f"![Key Risk Factors]({img_basename})\n\n"
    
    # if 'risk_breakdown' in analysis_result['visualization_paths']:
    #     # Use just the basename of the image
    #     #img_basename = os.path.basename(analysis_result['visualization_paths']['risk_breakdown'])
    #     #markdown_content += f"![Risk Contribution Breakdown]({img_basename})\n\n"
    
    # # Add suspicious patterns visualization if available
    # if 'suspicious_patterns' in analysis_result['visualization_paths']:
    #     markdown_content += "### Suspicious Patterns\n\n"
    #     # Use just the basename of the image
    #     img_basename = os.path.basename(analysis_result['visualization_paths']['suspicious_patterns'])
    #     markdown_content += f"![Suspicious Patterns]({img_basename})\n\n"
    
    # # Add risk severity distribution if available
    # if 'risk_severity' in analysis_result['visualization_paths']:
    #     markdown_content += "### Risk Severity Distribution\n\n"
    #     # Use just the basename of the image
    #     img_basename = os.path.basename(analysis_result['visualization_paths']['risk_severity'])
    #     markdown_content += f"![Risk Severity]({img_basename})\n\n"
    
    # Add feature contribution details in a table
    if 'shap_analysis' in analysis_result.get('explanations', {}):
        markdown_content += "### Top Risk Indicators\n\n"
        markdown_content += "| Factor | Impact | Direction |\n|--------|--------|---------|\n"
        
        contributors = analysis_result['explanations']['shap_analysis']['contributors']
        for contrib in contributors[:10]:  # Top 10 contributors
            # Add direction emoji
            direction_emoji = "🔼" if contrib['shap_value'] > 0 else "🔽"
            markdown_content += f"| {contrib['display_name']} | {abs(contrib['shap_value']):.4f} | {direction_emoji} {contrib['direction']} |\n"
    
    # Add GPT-4o mini explanation
    markdown_content += "\n## Expert Analysis\n\n"
    markdown_content += llm_explanation
    
    # Add timestamp
    markdown_content += f"\n\n---\n*Report generated on {timestamp}*"
    
    return markdown_content

def analyze_transaction_and_create_report(transaction_data, output_dir='./reports'):
    """
    Complete pipeline to analyze a transaction and create a markdown report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert dict to DataFrame if necessary
    if isinstance(transaction_data, dict):
        transaction_df = pd.DataFrame([transaction_data])
    else:
        transaction_df = transaction_data.copy()
    
    # Unique ID for this analysis (timestamp-based)
    analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a subdirectory for this specific analysis
    analysis_dir = os.path.join(output_dir, f"analysis_{analysis_id}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Step 1: Get detailed transaction analysis
    print(f"Analyzing transaction...")
    analysis_result = analyze_suspicious_transaction(
        transaction_df,
        model,
        output_dir=analysis_dir
    )
    
    # Step 2: Create prompt for GPT-4o 
    prompt = create_llm_prompt(analysis_result)
    
    # Step 3: Get explanation from GPT-4o 
   # Step 3: Get explanation from Gemini
    print(f"Generating expert explanation with Gemini...")
    llm_explanation = get_gemini_explanation(prompt) 
    
    # Step 4: Create comprehensive markdown report
    print(f"Creating markdown report...")
    markdown_content = create_markdown_report(
        analysis_result,
        llm_explanation
    )
    
    # Step 5: Save the report
    classification = "suspicious" if analysis_result['is_suspicious'] else "normal"
    report_filename = f"transaction_report_{classification}_{analysis_id}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Copy all generated images to the same directory as the report for easy access
    for key, img_path in analysis_result.get('visualization_paths', {}).items():
        if os.path.exists(img_path):
            try:
                import shutil
                dest_path = os.path.join(output_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dest_path)
                print(f"Copied image {os.path.basename(img_path)} to report directory: {dest_path}")
            except Exception as e:
                print(f"Error copying image {img_path}: {str(e)}")
    
    print(f"Report saved to: {report_path}")
    return report_path




