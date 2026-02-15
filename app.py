import os
from dotenv import load_dotenv

# 1. LOAD THIS FIRST
load_dotenv()

from src.agents import Predictor, Analyst, ComplianceAnalyst, CaseManager
from src.tasks import Compliance_Validation_Task, Prediction_task, Reporting_task, Case_creation_task
from crewai import Crew, Process
import streamlit as st
import glob
import datetime
import pandas as pd
import base64
import io
from PIL import Image
import re
import plotly.express as px
import matplotlib.pyplot as plt
import json


# File-based persistent storage for cases
CASES_FILE = "cases_data.json"

# Function to load cases from file
def load_cases():
    try:
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading cases: {e}")
        return []

# Function to save cases to file
def save_cases(cases):
    try:
        with open(CASES_FILE, 'w') as f:
            json.dump(cases, f, indent=2)
    except Exception as e:
        st.error(f"Error saving cases: {e}")

# Simpler function to get next case ID - based on count
def get_next_case_id():
    cases = load_cases()
    return f"C{len(cases) + 1:03d}"  # Format as C001, C002, etc.

# Function to create and update charts with updated risk levels
def create_priority_charts(cases):
    # Calculate analytics
    total_cases = len(cases)
    
    # Count priorities (case insensitive) with the new risk levels
    priority_counts = {
        "Critical": len([c for c in cases if str(c.get('priority', '')).lower() == "critical"]),
        "High": len([c for c in cases if str(c.get('priority', '')).lower() == "high"]),
        "Medium": len([c for c in cases if str(c.get('priority', '')).lower() == "medium"]),
        "Low": len([c for c in cases if str(c.get('priority', '')).lower() == "low"])
    }

    # Create DataFrame for the chart with proper order (Critical highest)
    chart_data = pd.DataFrame({
        "Priority": ["Critical", "High", "Medium", "Low"],
        "Count": [priority_counts["Critical"], priority_counts["High"], priority_counts["Medium"], priority_counts["Low"]]
    })
    
    # Return both the data and charts
    return {
        "total": total_cases,
        "counts": priority_counts,
        "chart_data": chart_data,
        "pie_chart": px.pie(
            chart_data,
            names='Priority',
            values='Count',
            title=f"Case Priority Distribution (Total: {total_cases})",
            color='Priority',
            color_discrete_map={
                'Critical': '#8E0000',  # Darker red for critical
                'High': '#FF6B6B',      # Red for high
                'Medium': '#FFD166',    # Yellow for medium
                'Low': '#4ECDC4'        # Teal for low
            }
        )
    }

# Function to display report content
# Function to display report content with better image handling
def display_report():
    # Create report directory if it doesn't exist
    if not os.path.exists("./report"):
        os.makedirs("./report")
    
    try:
        report_files = glob.glob("./report/*.md")
        
        if report_files:
            # Sort by modification time, most recent first
            latest_report = max(report_files, key=os.path.getmtime)
            report_dir = os.path.dirname(latest_report)
            
            # Read content with error handling
            try:
                with open(latest_report, 'r', encoding='utf-8', errors='replace') as f:
                    report_content = f.read()
                
                # Extract images from markdown and display them directly
                st.caption(f"Report from: {os.path.basename(latest_report)}")
                
                # Process report content by handling image references
                image_matches = re.findall(r'!\[(.*?)\]\((.*?)\)', report_content)
                
                # Prepare found and missing images
                found_images = []
                
                # Process each image reference
                for alt_text, img_path in image_matches:
                    # Normalize path separators and clean up the path
                    img_path = img_path.replace('\\', '/').strip()
                    if img_path.startswith('./'):
                        img_path = img_path[2:]
                        
                    # Try to locate the image file using multiple strategies
                    image_found = False
                    found_path = None
                    
                    # List of possible locations to search for the image
                    possible_paths = [
                        os.path.join(report_dir, img_path),  # Relative to report file
                        img_path,  # Exact path as specified
                        os.path.abspath(img_path),  # Absolute path
                    ]
                    
                    # Also check in analysis subdirectories of the report directory
                    analysis_dirs = glob.glob(os.path.join(report_dir, "analysis_*"))
                    if analysis_dirs:
                        # Sort by modification time to get the latest analysis directory first
                        latest_analysis_dir = max(analysis_dirs, key=os.path.getmtime)
                        possible_paths.append(os.path.join(latest_analysis_dir, os.path.basename(img_path)))
                        
                        # Try all analysis directories if needed
                        for analysis_dir in sorted(analysis_dirs, key=os.path.getmtime, reverse=True):
                            possible_paths.append(os.path.join(analysis_dir, os.path.basename(img_path)))
                    
                    # Try each path until we find the image
                    for test_path in possible_paths:
                        if os.path.exists(test_path):
                            try:
                                # Just check if it can be opened as an image without displaying yet
                                Image.open(test_path)
                                image_found = True
                                found_path = test_path
                                found_images.append({
                                    'alt_text': alt_text,
                                    'path': test_path
                                })
                                break
                            except Exception as e:
                                continue  # Try the next path
                
                # Remove image references from markdown content that couldn't be found
                # Replace ![alt](path) with nothing for missing images
                for alt_text, img_path in image_matches:
                    img_match_found = any(img['alt_text'] == alt_text for img in found_images)
                    if not img_match_found:
                        # Remove this image reference from the markdown
                        pattern = re.escape(f"![{alt_text}]({img_path})")
                        report_content = re.sub(pattern, "", report_content)
                
                # Display the cleaned markdown content
                st.markdown(report_content)
                
                # Display only the found images
                if found_images:
                    st.subheader("Report Images")
                    for img in found_images:
                        try:
                            image = Image.open(img['path'])
                            st.image(image, caption=img['alt_text'], use_column_width=True)
                        except Exception as e:
                            # If the image was found but still can't be displayed, don't show an error
                            pass
                
                # Provide download button
                with open(latest_report, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=os.path.basename(latest_report),
                        mime="text/markdown",
                    )
            except Exception as e:
                st.error(f"Error reading report: {str(e)}")
                st.info("Run compliance check to generate a report")
        else:
            st.info("No reports generated yet. Run compliance check to generate a report.")
    except Exception as e:
        st.error(f"Could not read report file: {str(e)}")
        st.info("Run compliance check to generate a report")
# Function to normalize priority values
def normalize_priority(priority):
    if not priority or not isinstance(priority, str):
        return "Medium"  # Default
    
    priority = priority.strip().lower()
    
    if priority in ['critical', 'c', 'crit']:
        return "Critical"
    elif priority in ['high', 'h']:
        return "High"
    elif priority in ['medium', 'm', 'mid']:
        return "Medium"
    elif priority in ['low', 'l']:
        return "Low"
    else:
        return "Medium"  # Default

# Page navigation using query params
current_page = st.query_params.get("page", "home")

# Display the appropriate page
if current_page == "cases":
    st.title("AML Cases Dashboard")
    
    # Add navigation link back to home
    st.markdown("[← Back to Compliance Checker](/?page=home)")
    
    # Display cases and analytics
    col1, col2 = st.columns([2, 1])
    
    # Always load fresh case data
    cases = load_cases()
    
    with col1:
        st.subheader("Recent Cases")
        if cases:
            # Add a refresh button
            if st.button("↻ Refresh Cases"):
                st.rerun()
                
            # Force refresh with a key based on total cases
            st.empty().text(f"Total cases: {len(cases)}")
                
            for case in sorted(cases, key=lambda x: x.get('created_at', ''), reverse=True):
                # Get priority with appropriate coloring
                priority = case.get('priority', 'Medium')
                priority_color = {
                    "Critical": "#8E0000",
                    "High": "#FF6B6B",
                    "Medium": "#FFD166",
                    "Low": "#4ECDC4"
                }.get(priority, "#808080")  # Default gray for unknown priority
                
                with st.expander(f"Case {case.get('case_id', 'Unknown')} - {priority} Priority"):
                    st.write(f"**Status:** {case.get('status', 'N/A')}")
                    st.write(f"**Risk Score:** {case.get('risk_score', 'N/A')}")
                    st.write(f"**Created:** {case.get('created_at', 'N/A')}")
                    
                    # Add a delete button for each case
                    if st.button("Delete Case", key=f"delete_{case.get('case_id')}"):
                        current_cases = load_cases()
                        updated_cases = [c for c in current_cases if c.get('case_id') != case.get('case_id')]
                        save_cases(updated_cases)
                        st.success(f"Case {case.get('case_id')} deleted")
                        st.rerun()
        else:
            st.info("No cases have been processed yet.")
    
    with col2:
        st.subheader("Case Analytics")
        if cases:
            # Generate charts using our helper function
            charts = create_priority_charts(cases)
            
            # Display counts
            st.metric("Total Cases", charts["total"])
            st.write("**Cases by Priority:**")
            st.write(f"- Critical: {charts['counts']['Critical']}")
            st.write(f"- High: {charts['counts']['High']}")
            st.write(f"- Medium: {charts['counts']['Medium']}")
            st.write(f"- Low: {charts['counts']['Low']}")
            
            # Display bar chart
            st.bar_chart(
                charts["chart_data"].set_index("Priority"),
                width='stretch',
                height=300
            )
            
            # Display pie chart
            st.plotly_chart(charts["pie_chart"], width='stretch')
            
            # Clear all cases button
            if st.button("Clear All Cases"):
                save_cases([])  # Save empty list to clear cases
                st.success("All cases cleared")
                st.rerun()
        else:
            st.info("No data available for analytics.")
else:  # Default to home page
    st.title("Transaction Anomaly Identifier")
    
    # Create a two-column layout for the main page
    main_col1, main_col2 = st.columns([3, 2])

    with main_col1:
        # Input form in the left column
        entity_name = st.text_input("Enter the entity name")
        entity_country = st.text_input("Enter the entity country")

        from_bank = st.number_input("Enter the sender bank code", min_value=0)
        account = st.text_input("Enter the sender account number")
        to_bank = st.number_input("Enter the receiver bank code", min_value=0)
        account_dest = st.text_input("Enter the receiver account number")
        amount_received = st.number_input("Enter the amount received in transaction", min_value=0.0)
        receiving_currency = st.text_input("Enter the currency of the received amount")
        amount_paid = st.number_input("Enter the amount paid in transaction", min_value=0.0)
        payment_currency = st.text_input("Enter the currency of the payment")
        payment_format = st.text_input("Enter the format of payment (e.g., Cheque, Wire)")
        date = st.text_input("Enter the transaction date (YYYY-MM-DD)")
        day = st.text_input("Enter the day of the week")
        time = st.text_input("Enter the transaction time (HH:MM:SS)")

        # Create tabs that are always visible
        tab1, tab2 = st.tabs(["Compliance Validation", "Report"])
        
        # Initialize state for the tabs
        if "validation_result" not in st.session_state:
            st.session_state.validation_result = None

        # Default content in tabs
        with tab1:
            st.subheader("Compliance Validation Results")
            if st.session_state.validation_result:
                st.markdown(st.session_state.validation_result)
            else:
                st.info("Compliance Check results will be displayed here")
        
        with tab2:
            st.subheader("Transaction Analysis Report")
            display_report()
        
        # Check Compliance button
        if st.button("Check Compliance"):
            if not entity_name or not entity_country or not from_bank or not account or not to_bank or not account_dest or not amount_received or not receiving_currency or not amount_paid or not payment_currency or not payment_format or not date or not day or not time:
                st.error("Please fill all the fields")
            else:
                with st.spinner("Checking compliance..."):            
                    compliance_validation = Compliance_Validation_Task(ComplianceAnalyst, entity_name, entity_country)
            
                    prediction = Prediction_task(
                        Predictor,
                        from_bank,
                        account,
                        to_bank,
                        account_dest,
                        amount_received,
                        receiving_currency,
                        amount_paid,
                        payment_currency,
                        payment_format,
                        date,
                        day,
                        time
                    )
                    
                    prediction_output = prediction.output

                    reporting_task = Reporting_task(Analyst, from_bank,
                        account,
                        to_bank,
                        account_dest,
                        amount_received,
                        receiving_currency,
                        amount_paid,
                        payment_currency,
                        payment_format,
                        date,
                        day,
                        time)

                    transaction_data = {
                        "from_bank": from_bank,
                        "account": account,
                        "to_bank": to_bank,
                        "account_dest": account_dest,
                        "amount_received": amount_received,
                        "receiving_currency": receiving_currency,
                        "amount_paid": amount_paid,
                        "payment_currency": payment_currency,
                        "payment_format": payment_format,
                        "date": date,
                        "day": day,
                        "time": time
                    }

                    case_creation = Case_creation_task(CaseManager, transaction_data)

                    crew = Crew(
                        agents=[Predictor, Analyst, ComplianceAnalyst, CaseManager],
                        tasks=[compliance_validation, prediction, reporting_task, case_creation],
                        process=Process.sequential,
                        verbose=True,
                        max_rpm=10,
                        memory=False
                    )

                    inputs = {
                        "from_bank": from_bank,
                        "account": account,
                        "to_bank": to_bank,
                        "account_dest": account_dest,
                        "amount_received": amount_received,
                        "receiving_currency": receiving_currency,
                        "amount_paid": amount_paid,
                        "payment_currency": payment_currency,
                        "payment_format": payment_format,
                        "date": date,
                        "day": day,
                        "time": time
                    }

                    crew.kickoff(inputs=inputs)
                    
                    # Update tab1 with compliance results
                    try:
                        compliance_result = compliance_validation.output
                        compliance_result = compliance_result.json_dict  # Extract dictionary

                        if isinstance(compliance_result, dict):
                            # Update risk level terminology if needed
                            if 'risk_level' in compliance_result:
                                compliance_result['risk_level'] = normalize_priority(compliance_result['risk_level'])
                                
                            formatted_result = f"""
                            **Status:** {compliance_result.get('status', 'N/A')}  
                            **Risk Level:** {compliance_result.get('risk_level', 'N/A')}  
                            **Details:** {compliance_result.get('details', 'No details provided')}  
                            **Recommendation:** {compliance_result.get('recommendation', 'No recommendation available')}  
                            """
                            
                            # Store in session state so it persists
                            st.session_state.validation_result = formatted_result
                        else:
                            st.session_state.validation_result = "⚠️ Output is not a dictionary, displaying raw results."
                    except Exception as e:
                        st.session_state.validation_result = f"❌ Error displaying results: {e}"
                    
                    # After successful processing, add case to the file-based storage
                    if 'case_creation' in locals() and hasattr(case_creation, 'output'):
                        try:
                            # Reload cases to ensure we have the latest data
                            current_cases = load_cases()
                            
                            # Get case from AI output
                            new_case = case_creation.output.json_dict
                            
                            # Explicitly set the case_id based on the number of cases
                            new_case['case_id'] = get_next_case_id()
                                
                            # Add timestamp for tracking
                            new_case['created_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Add status if missing
                            if 'status' not in new_case:
                                new_case['status'] = 'Open'
                                
                            # Normalize priority to use new categories
                            if 'priority' in new_case:
                                new_case['priority'] = normalize_priority(new_case['priority'])
                            
                            # Add to our loaded cases and save to file
                            current_cases.append(new_case)
                            save_cases(current_cases)
                            
                            st.success(f"Case {new_case.get('case_id')} created and saved to database")
                            
                            # Force refresh of dashboard - but only after showing success message
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding case to dashboard: {e}")

    # Right column - Dashboard - Always load fresh data
    with main_col2:
        st.header("Cases Dashboard")
        
        # Load cases directly from storage
        dashboard_cases = load_cases()
        
        # Show either real cases or an empty state
        if dashboard_cases:
            # Recent cases section
            st.subheader("Recent Cases")
            
            # Force refresh based on case count
            st.empty().text(f"Showing {min(3, len(dashboard_cases))} of {len(dashboard_cases)} cases")
            
            for case in sorted(dashboard_cases, key=lambda x: x.get('created_at', ''), reverse=True)[:3]:  # Show only 3 most recent
                # Get priority for display
                priority = case.get('priority', 'Medium')
                
                with st.expander(f"Case {case.get('case_id', 'Unknown')} - {priority} Priority"):
                    st.write(f"**Status:** {case.get('status', 'N/A')}")
                    st.write(f"**Risk Score:** {case.get('risk_score', 'N/A')}")
            
            # Analytics section
            st.subheader("Case Analytics")
            
            # Generate charts using our helper function for consistency
            charts = create_priority_charts(dashboard_cases)
            
            # Display pie chart
            st.plotly_chart(charts["pie_chart"], width='stretch')
            
            # Add a metric showing total cases
            st.metric("Total Cases", charts["total"])
            
            # Add view all cases button
            if st.button("View All Cases"):
                st.query_params["page"] = "cases"
                st.rerun()
        else:
            st.info("No cases have been processed yet. Submit a transaction to begin.")