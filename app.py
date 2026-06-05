import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Prediction", "📊 Historical Insights", "⛏️ CEC420 Mining Engine"])

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

with tab3:
    st.header("CEC420: Data Mining Discoveries (KDD)")
    st.markdown("Automated pattern extraction using Unsupervised Clustering and Dependency Modelling.")
    
    st.divider()

    # --- NEW: PREPROCESSING SUMMARY ---
    st.subheader("0. Data Preprocessing & Transformation")
    st.info("""
    **Transformation Applied:** One-Hot Encoding via `pd.get_dummies()`
    * **Purpose:** Machine learning algorithms require numerical input. The raw dataset contained categorical qualitative data (e.g., `Weather` = Rain, Clear, Dull).
    * **Action:** The system automatically transformed the `Weather` column into separate binary quantitative columns. 
    * **Result:** Non-numerical strings were successfully vectorized for mathematical processing without implying false ranking.
    """)
    st.divider()
    
    # --- 1. CLUSTERING (Unit 2 Requirement) ---
    st.subheader("1. Grid State Clustering (Unsupervised)")
    st.write("Discovering hidden structures in quantitative data (Voltage & Flicker) without using the known outage labels.")
    
    cluster_data = df[['Voltage_V', 'Flicker_Count']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=cluster_data, x='Voltage_V', y='Flicker_Count', hue='Cluster', palette='viridis', ax=ax1)
    plt.title("K-Means Clustering of Grid Behavior")
    st.pyplot(fig1)
    
    st.divider()

    # --- 2. RULE EXTRACTION (Unit 2 Requirement) ---
    st.subheader("2. Dependency Modelling & Rule Extraction")
    st.write("Extracting explicit logical relationships between categorical weather and quantitative voltage metrics.")
    
    X_tree = pd.get_dummies(df[['Voltage_V', 'Flicker_Count', 'Weather']], drop_first=True)
    y_tree = df['Outage_Occured']
    dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
    dtree.fit(X_tree, y_tree)
    
    tree_rules = export_text(dtree, feature_names=list(X_tree.columns))
    st.code(tree_rules, language='text')

    st.divider()

    # --- NEW: MODEL ACCURACY COMPARISON ---
    # --- 3. MODEL ACCURACY COMPARISON ---
    st.subheader("3. Model Performance Metrics")
    st.write("Benchmarking the foundational Logistic Regression model against the Decision Tree classifier.")
    
    # Train/Test Split (80/20)
    X_full = pd.get_dummies(df[['Voltage_V', 'Flicker_Count', 'Weather', 'ENEO_Post']], drop_first=True)
    y_full = df['Outage_Occured']
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    # Train both on the split
    lr_eval = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    dt_eval = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
    
    # Get Predictions
    lr_pred = lr_eval.predict(X_te)
    dt_pred = dt_eval.predict(X_te)
    lr_prob = lr_eval.predict_proba(X_te)[:, 1] # Needed for ROC-AUC
    
    # Generate Metrics DataFrame
    metrics_data = {
        "Model": ["Decision Tree (depth=3)", "Logistic Regression"],
        "Accuracy": [f"{accuracy_score(y_te, dt_pred)*100:.1f}%", f"{accuracy_score(y_te, lr_pred)*100:.1f}%"],
        "Precision": [f"{precision_score(y_te, dt_pred, zero_division=0):.2f}", f"{precision_score(y_te, lr_pred, zero_division=0):.2f}"],
        "Recall": [f"{recall_score(y_te, dt_pred, zero_division=0):.2f}", f"{recall_score(y_te, lr_pred, zero_division=0):.2f}"],
        "F1-Score": [f"{f1_score(y_te, dt_pred, zero_division=0):.2f}", f"{f1_score(y_te, lr_pred, zero_division=0):.2f}"],
        "ROC-AUC": ["—", f"{roc_auc_score(y_te, lr_prob):.2f}"]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display the table cleanly on the dashboard
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Train/Test Split for fair evaluation
    X_full = pd.get_dummies(df[['Voltage_V', 'Flicker_Count', 'Weather', 'ENEO_Post']], drop_first=True)
    y_full = df['Outage_Occured']
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    # Train both on the split
    lr_eval = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    dt_eval = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
    
    # Calculate Accuracy
    lr_acc = accuracy_score(y_te, lr_eval.predict(X_te)) * 100
    dt_acc = accuracy_score(y_te, dt_eval.predict(X_te)) * 100
    
    # Display Metrics
    comp1, comp2 = st.columns(2)
    comp1.metric("Logistic Regression Accuracy", f"{lr_acc:.1f}%")
    comp2.metric("Decision Tree Accuracy", f"{dt_acc:.1f}%", delta=f"{dt_acc - lr_acc:.1f}% vs LR", delta_color="normal")
    
    # Display Bar Chart
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=["Logistic Regression", "Decision Tree"], y=[lr_acc, dt_acc], palette="mako", ax=ax2)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    plt.title("Classification Performance")
    st.pyplot(fig2)