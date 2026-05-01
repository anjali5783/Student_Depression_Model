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

# --- 2. ULTIMATE CSS (Glassmorphism & Professional Brown Palette) ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #f5f1ed;
        border-right: 1px solid #e6d5c3;
    }
    .stApp {
        background: linear-gradient(to bottom, #fdfcfb, #e2d1c3);
    }
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(139, 69, 19, 0.1);
        margin-bottom: 20px;
    }
    /* Premium Button */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 3.5em;
        background: linear-gradient(45deg, #5D4037, #8B4513);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(93, 64, 55, 0.3);
    }
    h1, h2, h3 { color: #5D4037 !important; font-family: 'Poppins', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (Team Credits) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429153.png", width=100)
    st.title("Project Team")
    st.markdown("---")
    # Including your specific collaborators
    st.markdown("👨‍💻 **Lead Developer:** Anjali")
    st.markdown("👥 **Collaborators:**")
    st.write("• Abina")
    st.write("• Varsha")
    st.write("• Anna T.J.")
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
# Top Professional Banner
st.markdown(
    """
    <div style="width: 100%; border-radius: 20px; overflow: hidden; margin-bottom: 25px;">
        <img src="https://images.unsplash.com/photo-1523240715181-01489a943ee6?auto=format&fit=crop&q=80&w=2070" 
             style="width: 100%; height: 250px; object-fit: cover; filter: brightness(0.85);">
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>🎓 AI Student Depression Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #795548;'>Advanced machine learning for student mental health awareness.</p>", unsafe_allow_html=True)
st.divider()

if model:
    # Grouping inputs into visual cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Select...", "Male", "Female"], index=0)
        age = st.slider("Age", 18, 45, 18) 
        city = st.text_input("City", placeholder="Current City...")
        degree = st.text_input("Degree", placeholder="e.g. BCA")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📚 Academics")
        cgpa = st.number_input("CGPA", 0.0, 10.0, 0.0)
        academic_pressure = st.select_slider("Pressure", options=[1, 2, 3, 4, 5], value=1)
        study_sat = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=1)
        study_hours = st.number_input("Study Hours/Day", 0, 16, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧘 Lifestyle")
        sleep = st.number_input("Sleep (Hrs)", 0, 12, 0)
        diet = st.selectbox("Diet", ["Select...", "Healthy", "Moderate", "Unhealthy"], index=0)
        suicidal = st.selectbox("Suicidal Thoughts?", ["Select...", "No", "Yes"], index=0)
        fin_stress = st.select_slider("Financial Stress", options=[1, 2, 3, 4, 5], value=1)
        fam_history = st.selectbox("Family History", ["Select...", "No", "Yes"], index=0)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 6. PREDICTION ENGINE ---
    st.markdown("###")
    if st.button("🚀 INITIATE AI DIAGNOSTIC"):
        dropdowns = [gender, diet, suicidal, fam_history]
        if "Select..." in dropdowns:
            st.warning("Please complete all profile and lifestyle selections.")
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
                
                # Dynamic Results Area
                st.divider()
                res_col1, res_col2, res_col3 = st.columns([1,2,1])
                with res_col2:
                    st.markdown('<div class="glass-card" style="border: 2px solid #8B4513;">', unsafe_allow_html=True)
                    if prediction[0] == 1:
                        st.image("https://cdn-icons-png.flaticon.com/512/3593/3593451.png", width=120)
                        st.error("### ANALYSIS: HIGH RISK DETECTED")
                        st.write("Our AI suggests high vulnerability levels. We recommend reaching out to a mentor or professional guidance counselor.")
                    else:
                        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)
                        st.success("### ANALYSIS: LOW RISK DETECTED")
                        st.write("The AI analysis shows positive mental wellbeing metrics. Continue maintaining a healthy academic-life balance!")
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Diagnostic Error: {e}")