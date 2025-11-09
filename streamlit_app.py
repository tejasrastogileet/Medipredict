import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏥 Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🏥 MediPredict")
st.sidebar.write("Detect Early, Live Healthy")
st.sidebar.divider()

API_URL = "https://medipredict-ccsx.onrender.com"

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        st.sidebar.success("✅ API Connected")
        api_connected = True
    else:
        st.sidebar.warning("⚠️ API Error")
        api_connected = False
except:
    st.sidebar.error("❌ API Not Running")
    st.sidebar.error("Backend: Local Development")
    api_connected = False

st.sidebar.divider()
st.sidebar.info("""
**Backend Status:**

🔗 Deployed on Render
https://medipredict-ccsx.onrender.com

**Frontend:** Streamlit Cloud
🚀 Deployed & Live

**⏱️ Note:** First request takes ~50s
(Render free tier cold starts)
""")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Analytics", "ℹ️ About", "📚 Guide"])

with tab1:
    st.header("🔮 Disease Risk Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        disease = st.selectbox(
            "📋 Select Disease Type",
            ["diabetes", "heart", "liver", "kidney"],
            format_func=lambda x: {
                'diabetes': '🩺 Diabetes Prediction',
                'heart': '❤️ Heart Disease Detection',
                'liver': '🟠 Liver Disease Assessment',
                'kidney': '💧 Kidney Disease Risk'
            }[x]
        )
    
    with col2:
        st.metric("API Status", "🟢 Ready" if api_connected else "🔴 Down")
    
    st.divider()
    
    features = []
    feature_names = []
    
    if disease == "diabetes":
        st.write("**📋 Enter Patient Details:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input("👶 Pregnancies", 0, 17, 3, help="Number of times pregnant")
            glucose = st.number_input("🩸 Glucose (mg/dL)", 0, 300, 120, help="Plasma glucose concentration")
            blood_pressure = st.number_input("💓 Blood Pressure (mmHg)", 0, 200, 70, help="Diastolic blood pressure")
        
        with col2:
            skin_thickness = st.number_input("📏 Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
            insulin = st.number_input("🧬 Insulin (mU/ml)", 0, 900, 100, help="2-hour serum insulin")
            bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 25.0, step=0.1, help="Body mass index")
        
        with col3:
            diabetes_pedigree = st.number_input("👨‍👩‍👧 Diabetes Pedigree", 0.0, 3.0, 0.5, step=0.1, help="Diabetes family history")
            age = st.number_input("🎂 Age (years)", 20, 100, 35, help="Patient age")
        
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"]
    
    elif disease == "heart":
        st.write("**📋 Enter Patient Details:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("🎂 Age (years)", 20, 100, 50, help="Patient age")
            sex = st.selectbox("👥 Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0=Female, 1=Male")
            cp = st.number_input("🩹 Chest Pain Type (0-3)", 0, 3, 1, help="Type of chest pain")
            trestbps = st.number_input("💓 Resting BP (mmHg)", 80, 200, 120, help="Resting blood pressure")
        
        with col2:
            chol = st.number_input("🩸 Cholesterol (mg/dL)", 100, 600, 200, help="Serum cholesterol")
            fbs = st.selectbox("🍬 Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="0=No, 1=Yes")
            restecg = st.number_input("📊 Resting ECG (0-2)", 0, 2, 0, help="Resting electrocardiogram results")
        
        with col3:
            thalach = st.number_input("💨 Max Heart Rate", 60, 210, 150, help="Maximum heart rate achieved")
            exang = st.selectbox("🏃 Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="0=No, 1=Yes")
            oldpeak = st.number_input("📉 ST Depression", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
        
        with col4:
            slope = st.number_input("📈 Slope (0-2)", 0, 2, 1, help="Slope of peak exercise ST segment")
            ca = st.number_input("🔬 CA (0-4)", 0, 4, 0, help="Number of major vessels colored by fluoroscopy")
            thal = st.number_input("💓 Thal (0-3)", 0, 3, 2, help="Thalassemia: 0=normal, 1=fixed defect, 2=reversible defect")
        
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        feature_names = ["Age", "Sex", "ChestPain", "RestingBP", "Cholesterol", "FastingBloodSugar", "RestingECG", "MaxHeartRate", "ExerciseAngina", "STDepression", "Slope", "CA", "Thal"]
    
    elif disease == "liver":
        st.write("**📋 Enter Patient Details:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("🎂 Age (years)", 20, 100, 45, help="Patient age")
            gender = st.selectbox("👥 Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="0=Female, 1=Male")
            total_bilirubin = st.number_input("🔬 Total Bilirubin", 0.0, 10.0, 0.7, step=0.1, help="Total bilirubin level")
        
        with col2:
            direct_bilirubin = st.number_input("🔬 Direct Bilirubin", 0.0, 10.0, 0.2, step=0.1, help="Direct bilirubin level")
            alkaline_phosphatase = st.number_input("⚗️ Alkaline Phosphatase", 20, 500, 77, help="Alkaline phosphatase level")
            alamine_aminotransferase = st.number_input("⚗️ Alamine Aminotransferase (ALT)", 10, 500, 34, help="ALT enzyme level")
        
        with col3:
            aspartate_aminotransferase = st.number_input("⚗️ Aspartate Aminotransferase (AST)", 10, 500, 30, help="AST enzyme level")
            total_proteins = st.number_input("🧬 Total Proteins", 4.0, 10.0, 6.5, step=0.1, help="Total serum proteins")
        
        with col4:
            albumin = st.number_input("🧬 Albumin", 2.0, 5.0, 3.5, step=0.1, help="Albumin level")
            ag_ratio = st.number_input("🧬 A/G Ratio", 0.5, 2.0, 1.0, step=0.1, help="Albumin/Globulin Ratio")
        
        features = [age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, ag_ratio]
        feature_names = ["Age", "Gender", "TotalBilirubin", "DirectBilirubin", "AlkalinePhosphatase", "ALT", "AST", "TotalProteins", "Albumin", "A/G_Ratio"]
    
    elif disease == "kidney":
        st.write("**📋 Enter Patient Details:**")
        
        with st.expander("🔍 Basic Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("🎂 Age (years)", 2, 100, 50, help="Patient age")
                bp = st.number_input("💓 Blood Pressure (mmHg)", 50, 200, 80, help="Blood pressure")
                sg = st.number_input("🧪 Specific Gravity", 1.005, 1.030, 1.020, step=0.001, help="Urine specific gravity")
            with col2:
                al = st.number_input("🔬 Albumin (0-5)", 0, 5, 0, help="Albumin level in urine")
                su = st.number_input("🍬 Sugar (0-5)", 0, 5, 0, help="Sugar level in urine")
                rbc = st.selectbox("🩸 Red Blood Cells", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal", help="RBC status")
            with col3:
                pc = st.selectbox("🧬 Pus Cells", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Abnormal", help="Pus cells status")
                pcc = st.selectbox("🧬 Pus Cell Clumps", [0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present", help="Pus cell clumps")
                ba = st.selectbox("🦠 Bacteria", [0, 1], format_func=lambda x: "Not Present" if x == 0 else "Present", help="Bacteria presence")
        
        with st.expander("🔬 Blood Tests"):
            col1, col2, col3 = st.columns(3)
            with col1:
                bgr = st.number_input("🩸 Blood Glucose (mg/dL)", 70, 400, 120, help="Blood glucose random")
                bu = st.number_input("🔬 Blood Urea (mg/dL)", 10, 200, 40, help="Blood urea")
                sc = st.number_input("⚗️ Serum Creatinine (mg/dL)", 0.4, 10.0, 1.2, step=0.1, help="Serum creatinine")
            with col2:
                sod = st.number_input("🧂 Sodium (mEq/L)", 120, 145, 135, help="Sodium level")
                pot = st.number_input("🥔 Potassium (mEq/L)", 2.5, 8.0, 4.5, step=0.1, help="Potassium level")
                hemo = st.number_input("🩸 Hemoglobin (g/dL)", 3.0, 17.0, 13.0, step=0.1, help="Hemoglobin")
            with col3:
                pcv = st.number_input("📊 Packed Cell Volume", 9, 54, 40, help="Packed cell volume")
                wc = st.number_input("🦠 White Blood Cell Count (cells/cmm)", 2200, 26000, 8000, help="WBC count")
                rc = st.number_input("🔴 Red Blood Cell Count (millions/cmm)", 2.1, 8.0, 4.5, step=0.1, help="RBC count")
        
        with st.expander("🏥 Medical History"):
            col1, col2 = st.columns(2)
            with col1:
                htn = st.selectbox("💉 Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Hypertension")
                dm = st.selectbox("🩺 Diabetes Mellitus", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Diabetes mellitus")
                cad = st.selectbox("❤️ Coronary Artery Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="CAD")
            with col2:
                appet = st.selectbox("🍽️ Appetite", [0, 1], format_func=lambda x: "Good" if x == 0 else "Poor", help="Appetite")
                pe = st.selectbox("🦵 Pedal Edema", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help="Pedal edema")
        
        features = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe]
        feature_names = ["Age", "BP", "SpecificGravity", "Albumin", "Sugar", "RBC", "PusCells", "PusCellClumps", "Bacteria", "BloodGlucose", "BloodUrea", "SerumCreatinine", "Sodium", "Potassium", "Hemoglobin", "PCV", "WBC", "RBC_Count", "Hypertension", "DiabetesMellitus", "CAD", "Appetite", "PedalEdema"]
    
    st.divider()
    
    with st.expander("📊 Input Summary"):
        if len(features) > 0:
            summary_df = pd.DataFrame({"Feature": feature_names, "Value": features})
            st.dataframe(summary_df, use_container_width=True)
            st.info(f"✅ Total Features: {len(features)}")
        else:
            st.warning("⚠️ No features loaded")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Make Prediction", use_container_width=True, type="primary")
    
    if predict_button:
        if len(features) == 0:
            st.error("❌ No features to predict!")
        elif not api_connected:
            st.error("❌ API is not connected!")
            st.info("Make sure FastAPI backend is running: uvicorn app:app --reload")
        else:
            with st.spinner("🔄 Analyzing patient data..."):
                try:
                    time.sleep(1)
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"disease_type": disease, "features": features},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Prediction Complete!")
                        st.divider()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Disease Risk", f"{result['probability']:.1%}")
                        with col2:
                            st.metric("⚠️ Risk Level", result['risk_level'].split()[0])
                        with col3:
                            status = "🚨 Critical" if result['probability'] > 0.7 else ("⚠️ Alert" if result['probability'] > 0.4 else "✅ Safe")
                            st.metric("Status", status)
                        
                        st.divider()
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['probability'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"{disease.capitalize()} Risk (%)"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                            }
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.divider()
                        
                        if result['probability'] > 0.7:
                            st.error(f"🔴 {result['message']}")
                        elif result['probability'] > 0.4:
                            st.warning(f"🟡 {result['message']}")
                        else:
                            st.success(f"🟢 {result['message']}")
                        
                        st.divider()
                        st.write("**📋 Detailed Analysis:**")
                        analysis_col1, analysis_col2 = st.columns(2)
                        with analysis_col1:
                            st.write(f"**Disease Type:** {result['disease_type'].capitalize()}")
                            st.write(f"**Prediction:** {'Positive' if result['prediction'] == 1 else 'Negative'}")
                        with analysis_col2:
                            st.write(f"**Confidence:** {result['probability']:.1%}")
                            st.write(f"**Risk Category:** {result['risk_level']}")
                    else:
                        st.error(f"❌ Error: {response.json()['detail']}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API!")
                    st.info("⏱️ Render free tier takes ~50s to wake up. Please wait and try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("📊 Model Analytics & Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Model Performance Comparison")
        data = {'Model': ['Logistic\nRegression', 'Random\nForest', 'XGBoost', 'Gradient\nBoosting', 'MLP', 'Ensemble'], 'AUC-ROC': [0.88, 0.90, 0.91, 0.89, 0.87, 0.94]}
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Model', y='AUC-ROC', title='Model AUC-ROC Comparison', color='AUC-ROC', color_continuous_scale='Viridis', height=400)
        fig.add_hline(y=0.90, line_dash="dash", line_color="red", annotation_text="🎯 Threshold (0.90)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🏥 Disease Support Status")
        diseases_data = {'Disease': ['Diabetes', 'Heart', 'Liver', 'Kidney'], 'Status': [1, 1, 1, 1]}
        df_diseases = pd.DataFrame(diseases_data)
        fig = px.pie(df_diseases, names='Disease', values='Status', title='Diseases Supported', height=400, color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Overall AUC", "0.92+", "⭐ Excellent")
    with col2:
        st.metric("🤖 Models Used", "5", "Per Disease")
    with col3:
        st.metric("🏥 Diseases", "4", "Support")
    with col4:
        st.metric("📊 Avg Accuracy", "91%", "+2%")

with tab3:
    st.header("ℹ️ About This System")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        ### 🏥 MediPredict - Advanced Disease Prediction ML System
        **Detect Early, Live Healthy**
        
        **5-Model Ensemble:** Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Neural Networks
        
        **Performance:** AUC-ROC > 0.92 | Accuracy ~91% | Precision ~87% | Recall ~85%
        
        **Diseases:** 🩺 Diabetes (8) | ❤️ Heart (13) | 🟠 Liver (10) | 💧 Kidney (23)
        """)
    with col2:
        st.write("""
        ### ⚖️ Disclaimer
        ⚠️ **EDUCATIONAL ONLY**
        ❌ NOT for medical diagnosis
        ✅ Consult healthcare professionals
        """)

with tab4:
    st.header("📚 User Guide")
    st.write("""
    ### Setup
    **Terminal 1:** `uvicorn app:app --reload`
    **Terminal 2:** `streamlit run streamlit_app.py`
    
    ### Features by Disease
    - Diabetes: 8 features | Heart: 13 features | Liver: 10 features | Kidney: 23 features
    """)

st.divider()
footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])
with footer_col1:
    st.write("**🏥 MediPredict**\nv1.0.0")
with footer_col2:
    st.write("**Status:**\n✅ Operational" if api_connected else "❌ API Down")
with footer_col3:
    st.write("**Purpose:**\n🎓 ML Healthcare")