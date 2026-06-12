import streamlit as st
import os
from validator import ECommerceValidator

# 1. Setup the Web Page
st.set_page_config(page_title="AI Quality Gatekeeper", layout="wide")
st.title("🛍️ Automated E-Commerce Image Validator")
st.write("Upload a product image to verify if it meets our strict quality and AI standards.")

# 2. Load the AI Engine (Cached so it only loads once)
@st.cache_resource
def load_engine():
    return ECommerceValidator()

engine = load_engine()

# 3. Create the Upload Button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.divider()

    with st.spinner("AI is analyzing the image properties..."):
        report = engine.analyze_image(temp_path)

    # 4. Display the Results in Two Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scanned Image")
        st.image(report['output_file'], use_container_width=True)

    with col2:
        st.subheader("Quality Report")
        
        if "REJECTED" in report['status']:
            st.error(f"VERDICT: {report['status']}")
        else:
            st.success(f"VERDICT: {report['status']}")

        st.write("**Detected Objects:**")
        if report.get('detected'):
            st.info(", ".join(report['detected']).title())
        else:
            st.info("None")

        st.write("**Detailed Feedback:**")
        for reason in report.get('reasons', []):
            if "REJECTED" in report['status']:
                st.warning(f"⚠️ {reason}")
            else:
                st.success(f"✅ {reason}")

        st.divider() 
        
        # Display the Raw Metrics
        st.subheader("📊 Technical Image Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Resolution", report['metrics']['Resolution'])
        m2.metric("Sharpness", report['metrics']['Sharpness Score'])
        m3.metric("Brightness", report['metrics']['Brightness Score'])

        # Provide the Detailed Reason / Assessment
        st.write("**System Assessment:**")
        if "REJECTED" in report['status']:
            st.info("E-commerce platforms require images to be highly detailed, well-lit, and free of distracting backgrounds to ensure a premium customer experience. This image fell below our automated technical thresholds and must be retaken before publishing.")
        else:
            st.info("This image meets all technical baseline thresholds for professional e-commerce use. It is sharp, adequately lit, and the primary object is clearly identifiable.")

    if os.path.exists(temp_path):
        os.remove(temp_path)
