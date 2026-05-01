import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="Student Wellbeing AI",
    page_icon="🎓",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    /* This forces the banner to be wide but thin (rectangular) */
    .banner-img {
        width: 100%;
        height: 180px; /* Adjust this number to make it thinner or thicker */
        object-fit: cover; /* This crops the image to fit the rectangle perfectly */
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-card {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-top: 20px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background: linear-gradient(45deg, #ff4b4b, #ff7676);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    h1 { text-align: center; color: #2c3e50; }
    p { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
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
        st.error(f"Error loading files: {e}")
        return None, None, None

model, encoding_maps, global_mean = load_assets()

# --- 3. ENCODING HELPER ---
def encode_input(col, value):
    if not value or value.strip() == "":
        return global_mean
    if col in encoding_maps and value in encoding_maps[col]:
        return encoding_maps[col][value]
    return global_mean

# --- 4. TOP HERO SECTION ---
# Using HTML to force the "Rectangular" look
# --- 4. TOP HERO SECTION ---
# This uses inline styling to ensure it stretches and stays thin
# --- 4. TOP HERO SECTION ---
# Using a reliable, high-quality professional banner link
# --- 4. TOP HERO SECTION ---
st.markdown(
    """
    <div style="width: 100%; overflow: hidden; border-radius: 10px;">
        <img src="https://muntazirabidi.wordpress.com/wp-content/uploads/2022/12/black-white-and-gray-modern-professional-business-talk-linkedin-article-cover-image-4.png?w=1400" 
             style="width: 100%; height: 180px; object-fit: cover;">
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; margin-top: 15px;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identifying early markers of academic and emotional stress through Machine Learning.</p>", unsafe_allow_html=True)
st.divider()

# --- 5. UI INPUTS ---
if model:
    with st.sidebar:
        st.header("📊 Model Info")
        st.info("Algorithm: Random Forest")
        st.success("✅ Model Online")
        st.divider()
        st.write("© 2026 Student Wellbeing Project")

    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Profile")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 45, 20) 
            city = st.text_input("City", value="", placeholder="Current City...")
            degree = st.text_input("Degree", value="", placeholder="e.g. B.Tech")

        with col2:
            st.subheader("📚 Academics")
            cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
            academic_pressure = st.select_slider("Pressure", options=[1, 2, 3, 4, 5], value=3)
            study_sat = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=3)
            study_hours = st.number_input("Study Hours/Day", 0, 16, 6)

        with col3:
            st.subheader("🧘 Lifestyle")
            sleep = st.number_input("Sleep (Hrs)", 0, 12, 7)
            diet = st.selectbox("Diet", ["Moderate", "Healthy", "Unhealthy"])
            suicidal = st.selectbox("Suicidal Thoughts?", ["No", "Yes"])
            fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=2)
            fam_history = st.selectbox("Family History", ["No", "Yes"])

    st.markdown("###")
    
    # --- 6. PREDICTION LOGIC ---
    _, btn_col, _ = st.columns([1, 1, 1])
    
    with btn_col:
        if st.button("🚀 RUN AI DIAGNOSTIC"):
            try:
                feature_list = [
                    0, 1 if gender == "Male" else 0, age, encode_input("City", city),
                    encode_input("Profession", "Student"), academic_pressure, cgpa,
                    study_sat, sleep, encode_input("Dietary Habits", diet),
                    encode_input("Degree", degree), 1 if suicidal == "Yes" else 0,
                    study_hours, fin_stress, 1 if fam_history == "Yes" else 0
                ]

                input_array = np.array(feature_list).reshape(1, -1)
                prediction = model.predict(input_array)

                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                if prediction[0] == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=70)
                    st.error("### HIGH RISK DETECTED")
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/1484/1484947.png", width=70)
                    st.success("### LOW RISK DETECTED")
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
else:
    st.warning("Model assets missing.")