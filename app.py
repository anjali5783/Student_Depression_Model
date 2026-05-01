import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Student Wellbeing AI",
    page_icon="🎓",
    layout="wide"
)

# --- 2. PREMIUM BLACK THEME CSS ---
st.markdown("""
    <style>
    /* Deep Black Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Sidebar Dark Theme */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }

    /* Dark Glassmorphism Input Cards */
    .input-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* Text Color Styling */
    h1, h2, h3, p, label, .stMarkdown {
        color: #E6EDF3 !important;
    }

    /* Premium Blue Button */
    .stButton>button {
        background: linear-gradient(45deg, #1f6feb, #58a6ff) !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 111, 235, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    # High-quality thematic image for analysis
    st.image("https://images.unsplash.com/photo-1507413245164-6160d8298b31?q=80&w=2070&auto=format&fit=crop")
    st.markdown("### 📊 System Diagnostics")
    st.write("Assess student wellbeing metrics using automated machine learning analysis.")
    st.divider()
    st.caption("AI Diagnostic Tool v2.0")

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
# Displaying your specific "Mental Health" banner
try:
    banner = Image.open("image_4bbf61.jpg")
    st.image(banner, use_container_width=True)
except Exception as e:
    st.warning("Please ensure 'image_4bbf61.jpg' is in your project folder.")

st.markdown("<h1 style='text-align: center;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.divider()

if model:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"])
        age = st.slider("Age", 18, 45, 18) 
        city = st.text_input("City", placeholder="Enter city...")
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
        diet = st.selectbox("Diet", ["Select...", "Healthy", "Moderate", "Unhealthy"])
        suicidal = st.selectbox("Suicidal Thoughts?", ["Select...", "No", "Yes"])
        fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=1)
        fam_history = st.selectbox("Family History", ["Select...", "No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 INITIATE ANALYSIS"):
        if "Select..." in [gender, diet, suicidal, fam_history]:
            st.warning("Please complete all profile and lifestyle fields.")
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