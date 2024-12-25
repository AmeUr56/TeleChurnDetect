import streamlit as st
import streamlit.components.v1 as components

from numpy import array

from pipeline import pipeline
from joblib import load
from pathlib import Path

main_path = Path(__file__).parent.parent
model = load(main_path / "models/finetuned/voting_clf.joblib")

linkedin_profile_badge = """
<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="ameur-b-25a155247" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://dz.linkedin.com/in/ameur-b-25a155247?trk=profile-badge">Ameur B.</a></div>
"""

# Setup Page Configurations
st.set_page_config(
    page_title="TeleChurnDetect",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HTML/CSS for Columns Border 
st.markdown("""
    <style>
        /* Styling for columns to have subtle borders with rounded corners */
        .stColumn {
            border: 1px solid #e0e0e0;  /* Light gray border for a modern look */
            border-radius: 8px;  /* Rounded corners */
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Soft shadow for a modern effect */
            background-color: #fff;  /* White background */
        }
        
        /* Styling for rows to have subtle borders as well */
        .stBlock {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            background-color: #fff;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        /* Hover effect to make it interactive */
        .stColumn:hover, .stBlock:hover {
            border-color:rgb(16, 20, 16); /* Change border color to green on hover */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Slightly stronger shadow on hover */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("TeleChurnDetect")
st.sidebar.markdown("<a href='https://www.linkedin.com/in/ameur-b-25a155247/' target='_blank'><h1>Linkedin</h1></a>",unsafe_allow_html=True)
st.sidebar.markdown("<a href='https://x.com/Ame44i' target='_blank'><h1>X (Twitter)</h1></a>",unsafe_allow_html=True)

# Main Layout
st.title("TeleChurnDetect")
st.subheader("Churn Detection System for Telecom Companies")

st.warning("Empty values degrades system performance, try to fill all features.")


gender,senior_citizen,partner,dependents,tenure,phone_service = st.columns([2]*4+[3]+[2])
multiple_lines,internet_service,online_security,online_backup,device_protection,tech_support = st.columns([2]*6)
streamingtv,streaming_movies,contract,paperlessbilling,payment_method,monthly_charges,total_charges = st.columns([2]*7)

with st.form("form"):
    with gender:
        gender_val = st.selectbox(label="Gender",options=["Male","Female"])
    with senior_citizen:
        senior_citizen_val = "Yes" if st.checkbox(label="Is Senior Citizen?") else "No"
    with partner:
        partner_val = "Yes" if st.checkbox(label="Has Partner?") else "No"
    with dependents:
        dependents_val = "Yes" if st.checkbox(label="Has Dependents?") else "No"
    with tenure:
        tenure_val = st.slider(label="Tenure (in months)",min_value=0,max_value=120,step=1)
    with phone_service:
        phone_service_val = "Yes" if st.checkbox(label="Has Phone Service?") else "No"
    with multiple_lines:
        multiple_lines_val = "Yes" if st.checkbox(label="Has Phone Multiple Lines?") else "No"
    with internet_service:
        internet_service_val = st.selectbox(label="Internet Service",options=["No","DSL","Fiber optic"])
    with online_security:
        online_security_val = st.selectbox(label="Online Security",options=["No internet service","No","Yes"])
    with online_backup:
        online_backup_val = st.selectbox(label="Online Backup",options=["No internet service","No","Yes"])
    with device_protection:
        device_protection_val = st.selectbox(label="Device Protection",options=["No internet service","No","Yes"])
    with tech_support:
        tech_support_val = st.selectbox(label="Tech Support",options=["No internet service","No","Yes"])
    with streamingtv:
        streamingtv_val = st.selectbox(label="Streaming TV",options=["No internet service","No","Yes"])
    with streaming_movies:
        streaming_movies = st.selectbox(label="Streaming Movies",options=["No internet service","No","Yes"])
    with contract:
        contract_val = st.selectbox(label="Contract",options=["Month-to-month","One year","Two year"])
    with paperlessbilling:
        paperlessbilling_val = "Yes" if st.checkbox(label="Uses Paperless Billing") else "No"
    with payment_method:
        payment_method_val = st.selectbox(label="Payment Method",options=['Bank transfer (automatic)','Mailed check','Credit card (automatic)','Electronic check'])
    with monthly_charges:
        monthly_charges_val = st.number_input(label="Monthly Charges",min_value=0)
    with total_charges:
        total_charges_val = st.number_input(label="Total Charges",min_value=0)

    if st.form_submit_button(label="Detect"):
        df = array([gender_val,senior_citizen_val,partner_val,dependents_val,tenure_val,phone_service_val,multiple_lines_val,internet_service_val,online_security_val,
                                   online_backup_val,device_protection_val,tech_support_val,streamingtv_val,streaming_movies,contract_val,paperlessbilling_val,payment_method_val,monthly_charges_val,total_charges_val]).reshape((1,-1))
        df = pipeline(df)
        pred = "Churns" if model.predict(df) else "Stays"
        st.subheader(pred)