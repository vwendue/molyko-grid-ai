import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# 1. PAGE SETUP
st.set_page_config(page_title="Molyko Grid AI - Pro Dashboard", layout="wide")

# IMPROVED CSS FOR HIGH VISIBILITY
st.markdown("""
    <style>
    /* Main background */
    .main { background-color: #f0f2f6; }
    
    /* Metric Box Styling */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #d1d5db;
    }

    /* FORCING TEXT COLORS FOR VISIBILITY */
    /* Metric Label (The small text at the top of the box) */
    [data-testid="stMetricLabel"] {
        color: #1f2937 !important; 
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* Metric Value (The big number/text) */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    /* Metric Delta (The change indicator) */
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* General Markdown Text Color */
    .stMarkdown { color: #111827; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA & MODEL PREPARATION
@st.cache_data
def load_and_train():
    df = pd.read_csv('molyko_master_data.csv')
    # Train model
    X = pd.get_dummies(df[['Voltage_V', 'Flicker_Count', 'Weather', 'ENEO_Post']], drop_first=True)
    y = df['Outage_Occured']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return df, model, X.columns

df, model, model_columns = load_and_train()

# 3. SIDEBAR - INPUTS
st.sidebar.header("🛠️ Control Center")
input_voltage = st.sidebar.slider("Line Voltage (V)", 0, 240, 220)
input_flicker = st.sidebar.slider("Flicker Count (30m)", 0, 100, 5)
input_weather = st.sidebar.selectbox("Current Weather", ["Clear", "Dull", "Rain", "Heavy Wind"])
eneo_override = st.sidebar.toggle("🚨 ENEO Maintenance Announced")

# 4. MAIN LAYOUT
st.title("⚡ Molyko Smart Grid Predictor")
# Enhanced Data Status line
st.markdown(f"**DATA STATUS:** :blue[{len(df)} Records Indexed] | **LOCATION:** :blue[Molyko, Buea]")

st.divider()

# TOP METRICS (Now with fixed visibility)
m1, m2, m3, m4 = st.columns(4)
m1.metric("System Status", "MAINTENANCE" if eneo_override else "OPERATIONAL")
m2.metric("Line Voltage", f"{input_voltage}V", delta=f"{input_voltage-220}V")
m3.metric("Current Weather", input_weather)
m4.metric("Grid Flicker", f"{input_flicker} Hz")

st.divider()

# 5. RISK CALCULATION & TABS
tab1, tab2 = st.tabs(["🚀 Real-Time Prediction", "📊 Historical Insights"])

with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Blackout Probability")
        
        if eneo_override:
            risk_percent = 100.0
            st.error("### 🚨 ALERT: ENEO OFFICIAL OUTAGE")
            st.write("**Maintenance protocol active. Probability locked to 100% based on official ENEO report.**")
        else:
            input_data = pd.DataFrame([[input_voltage, input_flicker, input_weather, 0]], 
                                     columns=['Voltage_V', 'Flicker_Count', 'Weather', 'ENEO_Post'])
            input_encoded = pd.get_dummies(input_data).reindex(columns=model_columns, fill_value=0)
            risk_percent = model.predict_proba(input_encoded)[0][1] * 100
            
            if risk_percent > 80:
                st.error(f"## {risk_percent:.1f}% - EXTREME RISK")
            elif risk_percent > 40:
                st.warning(f"## {risk_percent:.1f}% - UNSTABLE")
            else:
                st.success(f"## {risk_percent:.1f}% - STABLE")

    with col_right:
        st.write("**Risk Visualizer**")
        st.progress(risk_percent / 100)
        st.info("The AI model estimates a " + ("High" if risk_percent > 50 else "Low") + " chance of immediate failure.")

with tab2:
    st.subheader("Ground Truth Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Weather-Induced Outage Probability**")
        weather_stats = df.groupby('Weather')['Outage_Occured'].mean() * 100
        st.bar_chart(weather_stats)
    with c2:
        st.write("**Factor Correlation Map**")
        fig, ax = plt.subplots()
        numeric_df = df[['Voltage_V', 'Flicker_Count', 'ENEO_Post', 'Outage_Occured']]
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)