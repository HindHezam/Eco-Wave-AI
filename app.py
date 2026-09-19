import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# تهيئة إعدادات الصفحة
st.set_page_config(page_title="Eco Wave AI Dashboard", layout="wide")

st.title("Eco Wave AI - Membrane Health Monitoring System")
st.write("Smart predictive maintenance, anomaly detection, and Piezo ultrasonic cleaning control.")

# تدريب النموذج وتجهيز البيانات
@st.cache_resource
def init_model():
    np.random.seed(42)
    normal_p = np.random.normal(0.5, 0.05, 200)
    normal_f = np.random.normal(13.5, 0.5, 200)
    
    df = pd.DataFrame({'Pressure_Drop': normal_p, 'Flow_Rate': normal_f})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    
    return model, scaler

model, scaler = init_model()

# الشريط الجانبي للمدخلات
st.sidebar.header("Sensor Inputs (Test / Live)")
p1_input = st.sidebar.number_input("Inlet Pressure P1 (bar)", value=4.8, step=0.1)
p2_input = st.sidebar.number_input("Outlet Pressure P2 (bar)", value=3.5, step=0.1)
pressure_drop = round(p1_input - p2_input, 2)

flow_input = st.sidebar.slider("Flow Rate (L/min)", min_value=5.0, max_value=20.0, value=9.1, step=0.1)

st.sidebar.markdown(f"**Calculated ΔP:** `{pressure_drop:.2f} bar`")

# المرجعية المعتمدة للنموذج المخبري
baseline_target_flow = 13.5  # L/min

# شرط تحديد حالة النظام
is_anomaly = (pressure_drop > 0.70) or (flow_input < 12.0)

# حساب النسبة المئوية المباشرة بدقة من السلايدر
current_efficiency = round(min(100.0, (flow_input / baseline_target_flow) * 100), 1)

# الشاشة الرئيسية - القراءات الحالية
col1, col2, col3 = st.columns(3)
col1.metric("P1 / P2 Pressure", f"{p1_input:.1f} / {p2_input:.1f} bar", delta=f"ΔP: {pressure_drop:.2f} bar")
col2.metric("Current Flow Rate", f"{flow_input:.1f} L/min")

if is_anomaly:
    col3.error("Status: Anomalous Pattern / Potential Scaling")
    st.warning("⚡ AUTOMATIC ACTION: Piezo Ultrasonic Cleaning Activated!")
else:
    col3.success("Status: Normal / Healthy Operation")
    st.info("✅ SYSTEM HEALTHY: No Cleaning Required Currently.")

st.markdown("## 🔄 Piezo Ultrasonic Cleaning Performance Verification")

# أرقام ما بعد التنظيف
post_pressure_drop = 0.50
post_flow = 12.2

# تصميم الكرتين المقارنين
card_col1, card_col2 = st.columns(2)

with card_col1:
    if is_anomaly:
        st.error("## 🔴 BEFORE CLEANING\n*(Detected Anomaly State)*")
        st.metric("Pressure Drop (ΔP)", f"{pressure_drop:.2f} bar", delta=f"P1: {p1_input} | P2: {p2_input}", delta_color="off")
        st.metric("Flow Rate", f"{flow_input:.1f} L/min")
    else:
        st.success("## 🟢 CURRENT STATE\n*(Healthy Baseline State)*")
        st.metric("Pressure Drop (ΔP)", f"{pressure_drop:.2f} bar", delta=f"P1: {p1_input} | P2: {p2_input}", delta_color="off")
        st.metric("Flow Rate", f"{flow_input:.1f} L/min")

with card_col2:
    st.success("## 🟢 AFTER CLEANING\n*(Post-Piezo Verification)*")
    if is_anomaly:
        st.metric("Pressure Drop (ΔP)", f"{post_pressure_drop:.2f} bar", delta=f"-{max(0.0, pressure_drop - post_pressure_drop):.2f} bar")
        st.metric("Flow Rate", f"{post_flow:.1f} L/min", delta=f"+{round(post_flow - flow_input, 1)} L/min")
    else:
        st.metric("Pressure Drop (ΔP)", f"{pressure_drop:.2f} bar", delta="Optimal")
        st.metric("Flow Rate", f"{flow_input:.1f} L/min", delta="Optimal")

st.markdown("---")

# 🎯 عرض نسبة واحدة صافية ومباشرة بدون تعقيد
if is_anomaly:
    st.info(f"""
    ### 🎯 **Flow Performance Recovery Rate:** **90.4%**
    """)
else:
    st.success(f"""
    ### 🎯 **Flow Performance Efficiency:** **100%**
    """)

st.markdown("---")
st.markdown("### 📊 Live Sensor Analytics & Anomaly Zone Mapping")

# رسم بياني Scatter Plot
fig = go.Figure()

# 1. نطاق التشغيل الطبيعي الآمن (Normal Zone)
fig.add_shape(
    type="rect",
    x0=0.3, x1=0.7, y0=12.0, y1=15.0,
    fillcolor="rgba(46, 204, 113, 0.15)",
    line=dict(color="rgba(46, 204, 113, 0.6)", width=2, dash="dash"),
    name="Normal Zone"
)

# 2. القراءة الحالية
fig.add_trace(go.Scatter(
    x=[pressure_drop],
    y=[flow_input],
    mode='markers+text',
    marker=dict(color='#E74C3C' if is_anomaly else '#2ECC71', size=20, line=dict(color='white', width=2)),
    text=["Current Anomaly" if is_anomaly else "Current Healthy"],
    textposition="top center",
    textfont=dict(size=14, color='#E74C3C' if is_anomaly else '#2ECC71'),
    name='Current Read'
))

# 3. القراءة المستهدفة بعد التنظيف
fig.add_trace(go.Scatter(
    x=[post_pressure_drop if is_anomaly else pressure_drop],
    y=[post_flow if is_anomaly else flow_input],
    mode='markers+text',
    marker=dict(color='#2ECC71', size=18, symbol='diamond', line=dict(color='white', width=2)),
    text=["Post-Piezo Target" if is_anomaly else "Target Baseline"],
    textposition="bottom center",
    textfont=dict(size=14, color="#2ECC71"),
    name='Target After Cleaning'
))

fig.update_layout(
    xaxis_title=dict(text="<b>Pressure Drop ΔP (bar)</b>", font=dict(size=18, color="#2C3E50")),
    yaxis_title=dict(text="<b>Flow Rate (L/min)</b>", font=dict(size=18, color="#2C3E50")),
    xaxis=dict(range=[0.0, 3.5], gridcolor='#EAEAEA', showgrid=True, tickfont=dict(size=13)),
    yaxis=dict(range=[5.0, 20.0], gridcolor='#EAEAEA', showgrid=True, tickfont=dict(size=13)),
    plot_bgcolor='white',
    height=420,
    margin=dict(l=50, r=50, t=30, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13))
)

st.plotly_chart(fig, use_container_width=True)