import streamlit as st
import pandas as pd
import joblib

# Pipelines
def pre_processing(data):
    X_variables = ['age', 'balance', 'day','duration', 'campaign', 'pdays','previous','has_loan','job__student',
                          'job__retired','job__blue-collar','job__entrepreneur','marital__single','marital__married','marital__divorced',
                          'edu_num','housing_loan','poutcome_num']
    return data[X_variables]

def post_processing(prediction):
    if len(prediction)==1:
        return prediction[:, 1][0]
    else:
        return prediction[:, 1]

def app_prediction_function(input_data, model):
    return post_processing(model.predict_proba(pre_processing(input_data)))
    
# Streamlit Web Interface    
st.header("Bank Marketing Web App")

# Inputs
age = st.number_input("Enter Age")
job__student=st.number_input("Student ? ")
job__retired=st.number_input("Retired ? ")
job__bluecollar = st.number_input("Bluecollar ? ")
job__entrepreneur = st.number_input("Entreprenueur ?")
marital__single=st.number_input("Single ")
marital__married=st.number_input("Married ")
marital__divorced = st.number_input("Divorced ")
edu_num = st.number_input("Education level")
balance = st.number_input("Average Yearly balance (in Euros)")
has_loan = st.number_input("Has personal loan ?")
housing_loan = st.number_input("Has housing loan ")
day = st.number_input("last contact day of the month")
duration=st.number_input("last contact duration (sec)")
campaign=st.number_input("No of contacts performed ? ")
pdays=st.number_input("Passed days after last contact date ")
previous = st.number_input("Previous contact times ")
poutcome_num = st.number_input("Outcome of previous campaign")

# Action button to initiate prediction 
if st.button("Predict"):
    
    # Load model
    model_file = 'model_rf3_bank.joblib'
    model = joblib.load(model_file)
    print(model)
    
    # Feature Dataset (row)
    input_data = pd.DataFrame([{'age':age, 'balance':balance, 'day':day,'duration':duration,'campaign':campaign, 'pdays':pdays,'previous':previous,'has_loan':has_loan,'job__student':job__student,
                          'job__retired':job__retired,'job__blue-collar':job__bluecollar,'job__entrepreneur':job__entrepreneur,'marital__single':marital__single,'marital__married':marital__married,'marital__divorced':marital__divorced,
                          'edu_num':edu_num,'housing_loan':housing_loan,'poutcome_num':poutcome_num}])
    
    # Predict
    prediction = app_prediction_function(input_data, model)
    
    # Output prediction
    st.text(f"Predicted Porbability: {prediction}")