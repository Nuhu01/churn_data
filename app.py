import streamlit as st
import pandas as pd
import joblib

# 1. Load the pre-trained model and preprocessors
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('label_encoder_gender.pkl')

# Assuming 'Geography' was encoded as 0 for France, 1 for Germany, 2 for Spain
# This mapping is derived from the order of unique values when fit_transform was called
# X['Geography'].unique() -> array(['France', 'Spain', 'Germany'], dtype=object)
# So, le.fit_transform(['France', 'Spain', 'Germany']) would be 0, 2, 1
# However, the previous notebook showed X['Geography'].unique() returning array(['France', 'Spain', 'Germany']),
# and then applying le.fit_transform(X['Geography']). This suggests the order might be alphabetical if not explicitly controlled.
# Let's assume the order observed from `X['Geography'].unique()` and then the order `fit_transform` assigns.
# Based on the previous output: array(['France', 'Spain', 'Germany'], dtype=object)
# If label encoder was applied on this series, the numerical mapping would be:
# France: 0, Germany: 1, Spain: 2 (if alphabetical)
# Or based on first appearance: France=0, Spain=1, Germany=2.
# Let's re-verify from the `le.classes_` attribute, but since we don't have it for Geography, 
# we will assume the order as seen in X['Geography'].unique() and map accordingly.
# The X['Geography'] was transformed in a separate cell, so let's try to infer it.
# From the notebook: `X['Geography'] = le.fit_transform(X['Geography'])` and `X['Geography'].unique()` gave `array(['France', 'Spain', 'Germany'])`.
# A new LabelEncoder instance was used for each. Let's assume the mapping for Geography is:
# France: 0, Germany: 1, Spain: 2 based on `le.fit_transform` usually assigning labels alphabetically if not explicitly specified.
# If `le_gender` was fitted on ['Female', 'Male'], then Female=0, Male=1.
# Let's create a dictionary for Geography mapping based on typical LabelEncoder behavior on alphabetical order.

geography_mapping = {'France': 0, 'Germany': 1, 'Spain': 2} 

# 2. Define prediction function
def predict_churn(features):
    # Create a DataFrame from the input features
    input_df = pd.DataFrame([features])

    # Preprocess 'Geography'
    input_df['Geography'] = geography_mapping[input_df['Geography'].iloc[0]]

    # Preprocess 'Gender' using the loaded LabelEncoder
    input_df['Gender'] = le_gender.transform(input_df['Gender'])

    # Select and scale numerical features
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Ensure the order of columns matches the training data
    # The 'selected_features' list from the notebook was:
    # ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    # We need to maintain this order.
    ordered_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    input_df = input_df[ordered_features]

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of churn (class 1)

    return prediction[0], prediction_proba[0]

# 3. Set up Streamlit application layout
st.set_page_config(page_title='Customer Churn Prediction App', layout='centered')
st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')

# 4. Create input widgets
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider('Credit Score', 350, 850, 600)
    geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))
    gender = st.selectbox('Gender', ('Female', 'Male'))
    age = st.slider('Age', 18, 92, 35)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)

with col2:
    balance = st.number_input('Balance', 0.0, 250000.0, 50000.0, step=1000.0)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('Is Active Member', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 100000.0, step=1000.0)

# Predict button
if st.button('Predict Churn'):
    user_input = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    churn_prediction, churn_probability = predict_churn(user_input)

    if churn_prediction == 1:
        st.error(f'The customer is likely to churn. (Probability: {churn_probability:.2f})')
    else:
        st.success(f'The customer is not likely to churn. (Probability: {churn_probability:.2f})')

