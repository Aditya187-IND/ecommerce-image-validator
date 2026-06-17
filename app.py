import streamlit as st
import os
import pandas as pd
import altair as alt
from validator import ECommerceValidator

st.set_page_config(page_title="AI Quality Gatekeeper", layout="wide")
st.title("🛍️ Automated E-Commerce Image Validator")
st.write("Upload a product image to verify if it meets our strict quality and AI standards.")

@st.cache_resource
def load_engine():
    return ECommerceValidator()

engine = load_engine()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.divider()

    with st.spinner("AI is executing deep image analysis..."):
        report = engine.analyze_image(temp_path)

    # --- TOP ROW: Image and Core Verdict ---
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Visual Analysis")
        
        # --- NEW: INTERACTIVE TABS FOR IMAGES ---
        tab1, tab2 = st.tabs(["🎯 AI Detections", "🩻 Structure X-Ray"])
        
        with tab1:
            st.image(report['output_file'], use_container_width=True, caption="YOLO Bounding Box Analysis")
        with tab2:
            st.image(report['blueprint_file'], use_container_width=True, caption="Canny Edge Geometry Extraction")

    with col2:
        st.subheader("Quality Gate Verdict")
        if "REJECTED" in report['status']:
            st.error(f"## {report['status']}")
        else:
            st.success(f"## {report['status']}")

        st.write("**System Checklist:**")
        for criteria, passed in report['checklist'].items():
            if passed:
                st.markdown(f"✅ **{criteria}**: Passed")
            else:
                st.markdown(f"❌ **{criteria}**: Failed")

        st.write("")
        st.write("**Detected Objects:**")
        if report.get('detected'):
            st.info(", ".join(report['detected']).title())
        else:
            st.info("None")

    st.divider() 

    if "REJECTED" in report['status'] and report.get('suggestions'):
        st.subheader("💡 Actionable Next Steps")
        st.write("Here is how you can fix the issues and get your image approved:")
        for suggestion in report['suggestions']:
            st.info(suggestion)
        st.divider()

    st.subheader("📊 Technical Telemetry & Color Distribution")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resolution", report['metrics']['Resolution'])
    m2.metric("Sharpness", report['metrics']['Sharpness'])
    m3.metric("Brightness", report['metrics']['Brightness'])
    m4.metric("File Size", report['metrics']['File Size'])

    st.write("")
    
    st.write("**Color Distribution Analysis:**")
    df = pd.DataFrame(report['colors'])
    color_chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=30).encode(
        x=alt.X('percent:Q', title='Percentage of Image (%)', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('hex:N', sort='-x', title='Hex Code', axis=alt.Axis(labelAngle=0)),
        color=alt.Color('hex:N', scale=None),
        tooltip=['hex', 'percent']
    ).properties(height=250)

    st.altair_chart(color_chart, use_container_width=True)

    st.write("")
    st.write("**Verbose System Assessment:**")
    for reason in report.get('reasons', []):
        if "REJECTED" in report['status']:
            st.warning(f"⚠️ {reason}")
        else:
            st.success(f"✅ {reason}")

    # Clean up both temp files
    if os.path.exists(temp_path):
        os.remove(temp_path)
