import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
# from st_aggrid import AgGrid

# Load model
model = joblib.load("heart_disease_model.pkl")

# Fetch dataset for analysis page
@st.cache_data
def load_data():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    y = y.iloc[:, 0].apply(lambda x: 1 if x > 0 else 0)
    df = pd.concat([X, y.rename("target")], axis=1)
    return df

data = load_data()

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Heart Disease App",
        ["Home", "Feature Info", "Predict", "Analysis & Tips"],
        icons=["heart", "info-circle", "activity", "bar-chart"],
        menu_icon="hospital",
        default_index=0
    )

# Page 1: Home with emoji/gif
if selected == "Home":
    st.image("https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif", width=300)
    st.title("Heart Disease Prediction using Machine Learning")
    st.markdown("""
        Welcome to the **Heart Disease Prediction App**! ❤️

        This tool uses a trained **Random Forest** model to predict the presence of heart disease based on key clinical features.

        Navigate through the app using the sidebar:
        - 🔎 Understand each feature
        - 🧠 Make your own prediction
        - 📊 Explore insights and prevention tips
    """)

# Page 2: Feature Info
elif selected == "Feature Info":
    st.header("🔎 Dataset Feature Information")

    st.markdown("""
    | Feature   | Description | Values/Encoding |
    |-----------|-------------|------------------|
    | age       | Age of the patient in years | Numeric (29–77) |
    | sex       | Gender of the patient | 0 = Female, 1 = Male |
    | cp        | Chest pain type | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
    | trestbps  | Resting blood pressure (mm Hg) | Numeric (94–200) |
    | chol      | Serum cholesterol (mg/dl) | Numeric (126–564) |
    | fbs       | Fasting blood sugar > 120 mg/dl | 0 = False, 1 = True |
    | restecg   | Resting ECG results | 0 = Normal, 1 = ST-T wave abnormality, 2 = Probable/Definite LVH |
    | thalach   | Maximum heart rate achieved | Numeric (71–202) |
    | exang     | Exercise-induced angina | 0 = No, 1 = Yes |
    | oldpeak   | ST depression induced by exercise | Numeric (0.0–6.2) |
    | slope     | Slope of the peak exercise ST segment | 0 = Upsloping, 1 = Flat, 2 = Downsloping |
    | ca        | Number of major vessels colored by fluoroscopy | 0–3 |
    | thal      | Thalassemia | 1 = Normal, 2 = Fixed defect, 3 = Reversible defect |
    """, unsafe_allow_html=True)

    st.subheader("🫀 Additional Medical Context")
    st.markdown("""
    - **Angina Types:**
        - *Typical angina*: Chest pain from reduced blood flow to the heart.
        - *Atypical angina*: Chest discomfort without classic symptoms.
        - *Non-anginal pain*: Chest pain not related to the heart.
        - *Asymptomatic*: No chest pain despite possible heart disease.

    - **Resting ECG Results:**
        - *0*: Normal.
        - *1*: ST-T wave abnormality (indicates ischemia).
        - *2*: Left Ventricular Hypertrophy (enlarged heart muscle).

    - **Slope of ST Segment:**
        - *0*: Upsloping – usually normal.
        - *1*: Flat – may suggest ischemia.
        - *2*: Downsloping – more strongly linked to ischemia.

    - **Thalassemia (Thal):**
        - *1*: Normal blood disorder status.
        - *2*: Fixed defect – permanent heart tissue damage.
        - *3*: Reversible defect – temporary blood flow issues.
    """)

# Page 3: Prediction
elif selected == "Predict":
    st.header("🧠 Heart Disease Prediction")

    st.markdown("**Enter patient details:**")
    age = st.slider("Age", 29, 77, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise-induced Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    user_input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    if st.button("Predict"):
        prediction = model.predict(user_input_df)[0]
        if prediction == 1:
            st.error("⚠️ The model predicts **presence** of heart disease.")
        else:
            st.success("✅ The model predicts **no heart disease**.")

# Page 4: Analysis & Prevention
elif selected == "Analysis & Tips":
    st.header("📊 Data Analysis & Heart Health Tips")
    st.subheader("1. Heart Disease Distribution")
    fig = px.pie(data, names='target', title='Heart Disease Distribution')
    st.plotly_chart(fig)

    st.subheader("2. Age vs Cholesterol")
    fig2 = px.scatter(data, x="age", y="chol", color="target",
                      labels={"target": "Heart Disease"})
    st.plotly_chart(fig2)

    st.subheader("3. Prevention Tips")
    st.markdown("""
    - 🏃‍♂️ **Exercise regularly** to maintain cardiovascular health
    - 🥗 **Eat a balanced diet** low in saturated fats
    - 🚭 **Avoid smoking** and limit alcohol
    - 🩺 **Monitor blood pressure** and cholesterol
    - 🧘‍♀️ **Manage stress** through meditation or yoga
    - 🧪 **Get regular checkups**, especially if you have risk factors
    """)

    st.subheader("4. Common Heart Diseases")
    st.markdown("""
    - ❤️ **Coronary Artery Disease (CAD)**: Blockage of arteries supplying blood to the heart muscle.
    - 💓 **Heart Attack (Myocardial Infarction)**: Complete blockage leading to damage of the heart muscle.
    - 💔 **Heart Failure**: The heart is unable to pump enough blood to meet the body’s needs.
    - 💓 **Arrhythmia**: Irregular heartbeat, either too fast or too slow.
    - 🫀 **Cardiomyopathy**: Disease of the heart muscle that makes it harder to pump blood.
    - 🫁 **Congenital Heart Disease**: Heart abnormalities present from birth.
    """)

    st.subheader("5. Common Causes of Heart Disease")
    st.markdown("""
    - 🍔 High cholesterol and poor diet
    - 🧂 High blood pressure (hypertension)
    - 🧬 Family history of heart disease
    - 🚬 Smoking
    - 💤 Lack of physical activity
    - 🧠 Stress and poor mental health
    """)
