import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        with open('titanic_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model not found. Please ensure 'titanic_model.pkl' exists.")
        return None

# Input preprocessing
def preprocess_input(data, label_encoders):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = int(data['FamilySize'] == 1)

    data['Sex_encoded'] = label_encoders['sex'].transform([data['Sex']])[0]
    data['Embarked_encoded'] = label_encoders['embarked'].transform([data['Embarked']])[0]

    if data['Sex'] == 'male':
        title = 'Mr' if data['Age'] >= 18 else 'Master'
    else:
        title = 'Mrs' if data['Age'] >= 18 else 'Miss'

    try:
        data['Title_encoded'] = label_encoders['title'].transform([title])[0]
    except ValueError:
        data['Title_encoded'] = 0

    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                'Fare', 'Embarked_encoded', 'FamilySize', 'IsAlone', 'Title_encoded']
    
    return np.array([[data[feature] for feature in features]])

# App title
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival probability:")

# Load model
model_data = load_model()
if model_data is None:
    st.stop()

# Input fields
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Gender", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children", 0, 10, 0)

with col2:
    fare = st.number_input("Ticket Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    st.write("**Port Info:** C = Cherbourg | Q = Queenstown | S = Southampton")

# Prediction logic
if st.button("Predict Survival", type="primary"):
    input_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }

    try:
        processed_input = preprocess_input(input_data, model_data['label_encoders'])
        model = model_data['model']

        if model_data['model_name'] != 'Random Forest':
            processed_input = model_data['scaler'].transform(processed_input)

        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0]

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("🎉 **Prediction: SURVIVED**")
            else:
                st.error("💀 **Prediction: DID NOT SURVIVE**")

        with col2:
            st.write("**Prediction Probabilities:**")
            st.write(f"Survival: {probability[1]:.2%}")
            st.write(f"Death: {probability[0]:.2%}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Survival Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


echo "# strimlit-assignment" >> README.md
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Madushi2/strimlit-assignment.git
git push -u origin main