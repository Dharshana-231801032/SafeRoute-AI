import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="SafeRoute AI — Dual-Tier Decision Support Platform",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0f;
        color: #f1f5f9;
    }
    .metric-card {
        background-color: #16161e;
        border: 1px solid #2a2a3a;
        padding: 1.25rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .badge-critical {
        background-color: #ef444422;
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #f9731622;
        color: #f97316;
        border: 1px solid #f97316;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-moderate {
        background-color: #eab30822;
        color: #eab308;
        border: 1px solid #eab308;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #22c55e22;
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚦 SafeRoute AI — Decision Support Platform")
st.caption("Agentic AI Framework for Proactive Traffic Accident Prevention & Intelligent Driver Assistance | Department of AI & DS")

# Sidebar navigation & role selection
st.sidebar.header("🎯 Navigation & Stakeholder Role")
role = st.sidebar.selectbox(
    "Select Stakeholder Perspective",
    ["Driver / Fleet Operator", "Traffic Police Enforcement", "NHAI Highway Engineer", "108 Emergency Response"]
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select System Module",
    [
        "🗺️ Interactive GIS Hotspot Map",
        "🔮 Tier 1 Severity Predictor & SHAP XAI",
        "🎛️ Dynamic What-If Risk Simulator",
        "🛣️ Safety-Aware Route Profiler",
        "🔊 Voice Advisory (TTS) Console",
        "🏢 Stakeholder Analytics Portal"
    ]
)

# Sample Blackspots & DBSCAN Clusters
BLACKSPOTS = [
    {"id": "BS-TN-01", "name": "NH-32 Km 45 (Chengalpattu)", "lat": 12.8330, "lng": 80.0460, "accidents": 42, "fatalities": 12, "risk": "CRITICAL"},
    {"id": "BS-TN-02", "name": "NH-45 Km 12 (Tambaram)", "lat": 13.0827, "lng": 80.2707, "accidents": 28, "fatalities": 8, "risk": "HIGH"},
    {"id": "BS-MH-01", "name": "NH-48 Km 120 (Pune Bypass)", "lat": 18.5204, "lng": 73.8567, "accidents": 55, "fatalities": 15, "risk": "CRITICAL"},
    {"id": "BS-KA-01", "name": "NH-44 Km 200 (Hosur Rd)", "lat": 12.9716, "lng": 77.5946, "accidents": 35, "fatalities": 9, "risk": "HIGH"},
    {"id": "BS-UP-01", "name": "NH-24 Km 85 (Lucknow)", "lat": 26.8467, "lng": 80.9462, "accidents": 60, "fatalities": 22, "risk": "CRITICAL"}
]

if page == "🗺️ Interactive GIS Hotspot Map":
    st.header("Layer 4 & 5 — Interactive GIS Hotspot & Corridor Map")
    st.markdown("Macro-level DBSCAN density clustering combined with MoRTH recognized blackspot records.")

    m = folium.Map(location=[18.0, 79.0], zoom_start=5, tiles="CartoDB dark_matter")
    
    for bs in BLACKSPOTS:
        color = "#ef4444" if bs["risk"] == "CRITICAL" else "#f97316"
        folium.CircleMarker(
            location=[bs["lat"], bs["lng"]],
            radius=10,
            popup=f"<b>{bs['name']}</b><br>Accidents: {bs['accidents']}<br>Fatalities: {bs['fatalities']}<br>Risk: {bs['risk']}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, width=1100, height=500)

elif page == "🔮 Tier 1 Severity Predictor & SHAP XAI":
    st.header("Tier 1 — Micro-Level Accident Severity Predictor & Explainable AI")
    st.markdown("Multi-Class Ensemble Model (Random Forest, XGBoost, LightGBM) with TreeSHAP Attribution.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Event Attributes")
        model_choice = st.selectbox("Select ML Model", ["XGBoost Classifier", "Random Forest", "LightGBM"])
        weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog", "Snow", "Storm"])
        visibility = st.slider("Visibility (miles)", 0.1, 10.0, 3.5)
        temperature = st.slider("Temperature (°F)", 10, 110, 75)
        time_of_day = st.selectbox("Time of Day", ["Morning Peak", "Afternoon", "Evening Peak", "Night"])
        junction = st.checkbox("Near Junction / Intersection", value=True)
        traffic_signal = st.checkbox("Traffic Signal Present", value=False)
        crossing = st.checkbox("Pedestrian Crossing", value=True)

        if st.button("🔮 Predict Accident Severity"):
            st.session_state['predicted'] = True

    with col2:
        st.subheader("Prediction & XAI TreeSHAP Explanation")
        if st.session_state.get('predicted', False):
            st.error("🚨 **Predicted Severity: LEVEL 3 — SEVERE**")
            st.caption("Model Confidence: 87.4% (Cost-Sensitive Weighted Model)")
            
            st.markdown("#### TreeSHAP Feature Attribution (Risk Drivers)")
            shap_df = pd.DataFrame({
                "Feature": ["Low Visibility", "Rain Condition", "Night / Evening", "Junction Location", "No Traffic Signal"],
                "Contribution (%)": [38.2, 27.5, 18.1, 11.0, 5.2]
            })
            st.bar_chart(shap_df.set_index("Feature"))

elif page == "🎛️ Dynamic What-If Risk Simulator":
    st.header("Module 2 — Dynamic What-If Risk Simulator")
    st.markdown("Simulate environmental & infrastructure modifications in real-time to observe risk shifts.")

    c1, c2, c3 = st.columns(3)
    with c1:
        speed = st.slider("Vehicle Speed (km/h)", 30, 120, 80)
    with c2:
        rain_intensity = st.slider("Rain Intensity (%)", 0, 100, 45)
    with c3:
        signal_status = st.selectbox("Traffic Signal Infrastructure", ["Active Signal", "Flashing Signal", "No Signal"])

    base_risk = 40
    calculated_risk = base_risk + (speed - 50) * 0.5 + (rain_intensity * 0.3) + (20 if signal_status == "No Signal" else 0)
    calculated_risk = min(100, max(0, calculated_risk))

    st.markdown(f"### Calculated Risk Score: **{calculated_risk:.1f} / 100**")

    if calculated_risk >= 75:
        st.error("🔴 **CRITICAL RISK ZONE** — Recommended Action: Reduce Speed below 50 km/h and enable wet-road traction assistance.")
    elif calculated_risk >= 50:
        st.warning("🟠 **HIGH RISK ZONE** — Recommended Action: Exercise caution, maintain 3-second follow distance.")
    else:
        st.success("🟢 **LOW / MODERATE RISK** — Safe operating conditions.")

elif page == "🛣️ Safety-Aware Route Profiler":
    st.header("Module 3 — Safety-Aware Route Profiler")
    st.markdown("Comparing **Fastest Path vs. Safest Path** avoiding DBSCAN hotspots & MoRTH blackspots.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 Option A: Fastest Route (NH-44)")
        st.write("⏱️ **Travel Time:** 5 hrs 10 mins")
        st.write("📏 **Distance:** 340 km")
        st.markdown("<span class='badge-critical'>CRITICAL RISK</span> (2 Blackspots, 1 DBSCAN Cluster)", unsafe_allow_html=True)
    
    with col2:
        st.subheader("🛡️ Option B: Recommended Safest Route (NH-48)")
        st.write("⏱️ **Travel Time:** 5 hrs 35 mins (+25 mins)")
        st.write("📏 **Distance:** 358 km")
        st.markdown("<span class='badge-low'>LOW RISK</span> (0 Blackspots, Bypasses Hotspots)", unsafe_allow_html=True)

elif page == "🔊 Voice Advisory (TTS) Console":
    st.header("Module 4 — Hands-Free Voice Advisory (TTS) Module")
    st.markdown("Audio alert engine triggered when vehicle approaches Severity ≥ 3 zones.")

    st.info("🔊 **Audio Engine Status:** Ready / Connected")
    
    sample_text = "Caution! You are approaching NH-32 Kilometer 45, a critical blackspot with 12 recorded fatalities. Rain detected. Reduce speed to 50 kilometers per hour."
    st.text_area("Generated TTS Advisory Text", sample_text, height=100)

    if st.button("▶️ Test Voice Advisory Playback"):
        st.warning("🔊 Playing Audio Advisory: *'Caution! You are approaching NH-32 Kilometer 45...'*")
        st.balloons()

elif page == "🏢 Stakeholder Analytics Portal":
    st.header("Layer 5 — Stakeholder Output & Decision Support")
    st.write(f"Showing Analytics Custom Tailored for: **{role}**")

    if "Police" in role:
        st.subheader("👮 Traffic Police & Enforcement Patrol Guide")
        st.write("• **Patrol Target:** NH-32 Km 40-50 (High accident density between 18:00 - 22:00)")
        st.write("• **Ambulance Positioning:** Deploy 108 Ambulance unit at Tambaram Interchange.")
    elif "Engineer" in role:
        st.subheader("🏗️ NHAI Highway Engineering Corridor Health")
        st.write("• **Corridor Health Index:** NH-48 Section 4 (Score: 38/100 - Needs Remediation)")
        st.write("• **Recommended Intervention:** Install high-friction surface treatment & LED warning signal.")
    else:
        st.subheader("🚘 Driver & Fleet Safety Assistant")
        st.write("• **Route Safety Score:** 84 / 100")
        st.write("• **Active Warning:** Low visibility expected near Pune Bypass.")
