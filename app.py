import streamlit as st
import pickle
import numpy as np
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Student Wellbeing AI",
    page_icon="🎓",
    layout="wide"
)

# --- 2. THE ULTIMATE VISIBILITY CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #fdfbfb;
    }
    
    /* Sidebar Styling - Solid Background & Dark Text */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #d7ccc8;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2 {
        color: #3e2723 !important;
    }

    /* Solid White Input Cards for Contrast */
    .input-card {
        background: #ffffff !important;
        border-radius: 15px;
        padding: 25px;
        border: 2px solid #d7ccc8;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Forced Dark Brown Text for Visibility */
    p, label, .stMarkdown, .stSubheader, h1, h2, h3, div, span {
        color: #3e2723 !important;
        font-weight: 600 !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #5d4037, #8d6e63) !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: none;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (Restored) ---
with st.sidebar:
    st.markdown("## 📚 Project Team")
    st.markdown("---")
    st.markdown("👨‍💻 **Lead Developer:** Anjali V.S")
    st.markdown("👥 **Collaborator:**")
    st.write("• Jani A P")
    st.markdown("---")
    st.info("BCA Final Year Project - 2026")

# --- 4. ASSET LOADING ---
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
        st.error(f"Asset Error: {e}")
        return None, None, None

model, encoding_maps, global_mean = load_assets()

def encode_input(col, value):
    if value == "Select..." or not value: return global_mean
    return encoding_maps.get(col, {}).get(value, global_mean)

# --- 5. MAIN UI ---
# Reliable Banner Image
st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=2070&auto=format&fit=crop", use_container_width=True)

st.markdown("<h1 style='text-align: center;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.divider()

if model:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"])
        age = st.slider("Age", 18, 45, 18) 
        city = st.text_input("City", placeholder="e.g. Cochin")
        degree = st.text_input("Degree", placeholder="e.g. BCA")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("📚 Academics")
        cgpa = st.number_input("CGPA", 0.0, 10.0, 0.0)
        academic_pressure = st.select_slider("Pressure", options=[1, 2, 3, 4, 5], value=1)
        study_sat = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=1)
        study_hours = st.number_input("Study Hours/Day", 0, 16, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🧘 Lifestyle")
        sleep = st.number_input("Sleep (Hrs)", 0, 12, 0)
        diet = st.selectbox("Dietary Habits", ["Select...", "Healthy", "Moderate", "Unhealthy"])
        suicidal = st.selectbox("Suicidal Thoughts?", ["Select...", "No", "Yes"])
        fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=1)
        fam_history = st.selectbox("Family History", ["Select...", "No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 RUN AI DIAGNOSTIC"):
        if "Select..." in [gender, diet, suicidal, fam_history]:
            st.warning("⚠️ Please fill out all selections before running.")
        else:
            try:
                features = [
                    0, 1 if gender == "Male" else 0, age, encode_input("City", city),
                    encode_input("Profession", "Student"), academic_pressure, cgpa,
                    study_sat, sleep, encode_input("Dietary Habits", diet),
                    encode_input("Degree", degree), 1 if suicidal == "Yes" else 0,
                    study_hours, fin_stress, 1 if fam_history == "Yes" else 0
                ]
                prediction = model.predict(np.array(features).reshape(1, -1))
                
                st.divider()
                if prediction[0] == 1:
                    st.error("### ⚠️ ANALYSIS: HIGH RISK DETECTED")
                else:
                    st.success("### ✅ ANALYSIS: LOW RISK DETECTED")
                    st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")