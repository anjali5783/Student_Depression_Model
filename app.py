import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Student Wellbeing AI", layout="wide")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("encoding_maps.pkl", "rb") as f:
            encoding_maps = pickle.load(f)
        with open("global_mean.pkl", "rb") as f:
            global_mean = pickle.load(f)
        return model, encoding_maps, global_mean
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, encoding_maps, global_mean = load_assets()

def encode_input(col, value):
    if not value or str(value).strip() == "" or value == "Select...": 
        return global_mean
    return encoding_maps.get(col, {}).get(value, global_mean)

# --- 3. UI LAYOUT ---
st.markdown("<h1 style='text-align: center;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.divider()

if model:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"], index=0)
        # AGE: Starts at the very beginning (18)
        age = st.slider("Age", 18, 45, 18) 
        city = st.text_input("City", placeholder="Current City...")
        degree = st.text_input("Degree", placeholder="e.g. B.Tech")

    with col2:
        st.subheader("📚 Academics")
        cgpa = st.number_input("CGPA", 0.0, 10.0, 0.0)
        # PRESSURE: Starts at the very beginning (1)
        academic_pressure = st.select_slider("Pressure", options=[1, 2, 3, 4, 5], value=1)
        # SATISFACTION: Starts at the very beginning (1)
        study_sat = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=1)
        study_hours = st.number_input("Study Hours/Day", 0, 16, 4)

    with col3:
        st.subheader("🧘 Lifestyle")
        sleep = st.number_input("Sleep (Hrs)", 0, 12, 8)
        diet = st.selectbox("Diet", ["Select...", "Healthy", "Moderate", "Unhealthy"], index=0)
        suicidal = st.selectbox("Suicidal Thoughts?", ["Select...", "No", "Yes"], index=0)
        # FINANCIAL STRESS: Starts at the very beginning (1)
        fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=1)
        fam_history = st.selectbox("Family History", ["Select...", "No", "Yes"], index=0)

    # --- 4. PREDICTION ---
    st.markdown("###")
    if st.button("🚀 RUN AI DIAGNOSTIC", use_container_width=True):
        # Validation check for dropdowns
        if "Select..." in [gender, diet, suicidal]:
            st.warning("Please complete all selections (Gender, Diet, and Thoughts).")
        else:
            try:
                feature_list = [
                    0, 1 if gender == "Male" else 0, age, encode_input("City", city),
                    encode_input("Profession", "Student"), academic_pressure, cgpa,
                    study_sat, sleep, encode_input("Dietary Habits", diet),
                    encode_input("Degree", degree), 1 if suicidal == "Yes" else 0,
                    study_hours, fin_stress, 1 if fam_history == "Yes" else 0
                ]
                prediction = model.predict(np.array(feature_list).reshape(1, -1))
                
                if prediction[0] == 1:
                    st.error("### HIGH RISK DETECTED")
                else:
                    st.success("### LOW RISK DETECTED")
            except Exception as e:
                st.error(f"Prediction Error: {e}")