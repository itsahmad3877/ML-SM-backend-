import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# --- PAGE SETUP ---
st.set_page_config(page_title="Digital Wellness Analytics", layout="centered")

st.title("📊 Digital Wellness Analytics Platform")
st.subheader("Social Media User Behavior Predictive Engine")
st.markdown("---")

# --- LOAD MODELS ---
CLASSIFICATION_PATH = 'best_classification_behavior_model.pkl'
REGRESSION_PATH = 'best_regression_behavior_model.pkl'

@st.cache_resource
def load_models():
    class_pack, reg_pack = None, None
    if os.path.exists(CLASSIFICATION_PATH):
        with open(CLASSIFICATION_PATH, 'rb') as f:
            class_pack = pickle.load(f)
    if os.path.exists(REGRESSION_PATH):
        with open(REGRESSION_PATH, 'rb') as f:
            reg_pack = pickle.load(f)
    return class_pack, reg_pack

class_pack, reg_pack = load_models()

# --- HELPER FUNCTION FOR DATA PROCESSING ---
def transform_inputs_to_blueprint(raw_inputs, expected_blueprint):
    raw_df = pd.DataFrame([raw_inputs])
    
    # Custom Feature Engineering
    raw_df['engagement_density_ratio'] = raw_inputs['likes_given_per_day'] / (raw_inputs['posts_per_week'] + 1)
    raw_df['session_intensity'] = raw_inputs['avg_session_duration_min'] * raw_inputs['sessions_per_day']

    categorical_columns = ['gender', 'country', 'profession', 'primary_platform', 
                           'preferred_device', 'peak_usage_time', 'sleep_disruption', 'mood_while_scrolling']
    
    processed_features = {}
    for column in raw_df.columns:
        if column not in categorical_columns:
            processed_features[column] = float(raw_df[column].iloc[0])

    for col in categorical_columns:
        supplied_val = str(raw_df[col].iloc[0])
        processed_features[f"{col}_{supplied_val}"] = 1.0

    final_aligned_vector = [processed_features.get(f, 0.0) for f in expected_blueprint]
    return np.array([final_aligned_vector])

# --- USER INTERFACE CONFIGURATION PANEL ---
st.sidebar.header("👥 Project Metadata")
st.sidebar.markdown("**Submitted To:**\n* Ma'am Andleeb\n* Ma'am Hamna")
st.sidebar.markdown("**Team Members:**\n* Syed Musa Elahi\n* Muhammad Ahmad\n* Hamna Khalil")

st.markdown("### 📝 Enter User Behavioral Attributes")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=22)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
    country = st.selectbox("Country", ["Pakistan", "India", "USA"])

with col2:
    profession = st.selectbox("Profession", ["Student", "Professional"])
    primary_platform = st.selectbox("Primary Platform", ["Instagram", "TikTok", "YouTube"])
    sessions_per_day = st.number_input("Daily Log-in Sessions", min_value=1, value=6)

with col3:
    avg_session_duration_min = st.number_input("Avg Session Duration (Mins)", min_value=1, value=45)
    platforms_used_count = st.number_input("Platforms Active Count", min_value=1, value=3)
    likes_given_per_day = st.number_input("Likes Per Day", min_value=0, value=85)

col4, col5, col6 = st.columns(3)
with col4:
    posts_per_week = st.number_input("Posts Per Week", min_value=0, value=2)
    self_reported_mental_health_score = st.slider("Mental Health Score (1-10)", 1.0, 10.0, 6.5, step=0.1)

with col5:
    sleep_disruption = st.selectbox("Sleep Disruption", ["No impact", "Mild impact", "Moderate impact"])
    peak_usage_time = st.selectbox("Peak Usage Time", ["Morning (6-10am)", "Evening (4-8pm)", "Night (8-12pm)"])

with col6:
    mood_while_scrolling = st.selectbox("Mood While Scrolling", ["Happy", "Neutral", "Bored"])
    preferred_device = st.selectbox("Preferred Device", ["Smartphone", "Laptop"])

# Map UI choices into a structured library bundle
input_data = {
    'age': age, 'gender': gender, 'country': country, 'profession': profession,
    'primary_platform': primary_platform, 'platforms_used_count': platforms_used_count,
    'sessions_per_day': sessions_per_day, 'avg_session_duration_min': avg_session_duration_min,
    'preferred_device': preferred_device, 'likes_given_per_day': likes_given_per_day,
    'posts_per_week': posts_per_week, 'self_reported_mental_health_score': self_reported_mental_health_score,
    'sleep_disruption': sleep_disruption, 'peak_usage_time': peak_usage_time, 'mood_while_scrolling': mood_while_scrolling
}

st.markdown("---")

# --- CALCULATION LOGIC ENGINES ---
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("🔮 Predict Concern Level", type="primary", use_container_width=True):
        if class_pack is not None:
            blueprint = class_pack['selected_features_blueprint']
            matrix = transform_inputs_to_blueprint(input_data, blueprint)
            scaled_x = class_pack['scaler'].transform(matrix)
            pred_idx = int(class_pack['model'].predict(scaled_x)[0])
            label = class_pack['class_labels'][pred_idx]
            st.success(f"**Classification Result:** {label}")
        else:
            st.error("Classification model file missing in repository.")

with btn_col2:
    if st.button("📈 Forecast Screen Time Hours", type="secondary", use_container_width=True):
        if reg_pack is not None:
            blueprint = reg_pack['selected_features_blueprint']
            matrix = transform_inputs_to_blueprint(input_data, blueprint)
            scaled_x = reg_pack['scaler'].transform(matrix)
            pred_val = float(reg_pack['model'].predict(scaled_x)[0])
            hours = round(max(0.0, pred_val), 2)
            st.info(f"**Regression Forecast:** {hours} Hours/Day")
        else:
            st.error("Regression model file missing in repository.")
