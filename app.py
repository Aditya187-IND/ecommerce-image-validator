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
    # Save the uploaded file temporarily so our OpenCV engine can read it
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.divider() # Adds a nice visual line

    with st.spinner("AI is analyzing the image properties..."):
        # Run your existing backend engine!
        report = engine.analyze_image(temp_path)

    # 4. Display the Results in Two Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scanned Image")
        # Display the image with the AI bounding boxes drawn on it
        st.image(report['output_file'], use_container_width=True)

    with col2:
        st.subheader("Quality Report")
        
        # Display a massive Pass/Fail badge
        if "REJECTED" in report['status']:
            st.error(f"VERDICT: {report['status']}")
        else:
            st.success(f"VERDICT: {report['status']}")

        st.write("**Detected Objects:**")
        if report.get('detected'):
            st.info(", ".join(report['detected']).title())
        else:
            st.info("None")

        st.write("**Detailed Feedback & Properties:**")
        for reason in report.get('reasons', []):
            # FIXED: Standard if/else block to prevent DeltaGenerator printout
            if "REJECTED" in report['status']:
                st.warning(f"⚠️ {reason}")
            else:
                st.success(f"✅ {reason}")

    # Clean up the temporary file after displaying
    if os.path.exists(temp_path):
        os.remove(temp_path)
