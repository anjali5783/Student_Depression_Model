import streamlit as st
import pickle
import numpy as np
import os

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="Student Wellbeing AI",
    page_icon="🎓",
    layout="wide"
)

# --- 2. PROFESSIONAL STYLING (Glassmorphism & Brown Gradients) ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f1ed 0%, #e6d5c3 100%);
    }
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(139, 69, 19, 0.2);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(45deg, #8B4513, #A0522D);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.4s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(139, 69, 19, 0.4);
    }
    h1, h2, h3 { color: #5D4037 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING ---
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

# --- 4. BANNER IMAGE ---
st.markdown(
    """
    <div style="width: 100%; overflow: hidden; border-radius: 15px; margin-bottom: 20px;">
        <img src="https://muntazirabidi.wordpress.com/wp-content/uploads/2022/12/black-white-and-gray-modern-professional-business-talk-linkedin-article-cover-image-4.png?w=1400" 
             style="width: 100%; height: 200px; object-fit: cover;">
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.divider()

# --- 5. UI INPUTS (All starting from beginning/zero) ---
if model:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"], index=0)
        age = st.slider("Age", 18, 45, 18) 
        city = st.text_input("City", placeholder="Current City...")
        degree = st.text_input("Degree", placeholder="e.g. B.Tech")

    with col2:
        st.subheader("📚 Academics")
        cgpa = st.number_input("CGPA", 0.0, 10.0, 0.0)
        academic_pressure = st.select_slider("Pressure", options=[1, 2, 3, 4, 5], value=1)
        study_sat = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=1)
        # CHANGED TO ZERO
        study_hours = st.number_input("Study Hours/Day", 0, 16, 0)

    with col3:
        st.subheader("🧘 Lifestyle")
        # CHANGED TO ZERO
        sleep = st.number_input("Sleep (Hrs)", 0, 12, 0)
        diet = st.selectbox("Diet", ["Select...", "Healthy", "Moderate", "Unhealthy"], index=0)
        suicidal = st.selectbox("Suicidal Thoughts?", ["Select...", "No", "Yes"], index=0)
        fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=1)
        fam_history = st.selectbox("Family History", ["Select...", "No", "Yes"], index=0)

    st.markdown("###")
    
    # --- 6. PREDICTION LOGIC ---
    if st.button("🚀 RUN AI DIAGNOSTIC"):
        dropdowns = [gender, diet, suicidal, fam_history]
        if "Select..." in dropdowns:
            st.warning("Please complete all selections (Gender, Diet, Thoughts, and History).")
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
                
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                if prediction[0] == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=80)
                    st.error("### HIGH RISK DETECTED")
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/1484/1484947.png", width=80)
                    st.success("### LOW RISK DETECTED")
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")