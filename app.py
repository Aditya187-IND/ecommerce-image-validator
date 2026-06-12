import streamlit as st
import os
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
    col1, col2 = st.columns([1.5, 1]) # Makes the image column slightly wider

    with col1:
        st.subheader("Visual Analysis")
        st.image(report['output_file'], use_container_width=True)

    with col2:
        st.subheader("Quality Gate Verdict")
        if "REJECTED" in report['status']:
            st.error(f"## {report['status']}")
        else:
            st.success(f"## {report['status']}")

        # --- THE VISUAL CHECKLIST ---
        st.write("**System Checklist:**")
        for criteria, passed in report['checklist'].items():
            if passed:
                st.markdown(f"✅ **{criteria}**: Passed")
            else:
                st.markdown(f"❌ **{criteria}**: Failed")

        st.write("") # spacing
        st.write("**Detected Objects:**")
        if report.get('detected'):
            st.info(", ".join(report['detected']).title())
        else:
            st.info("None")

    st.divider() 

    # --- BOTTOM ROW: Data Dashboard ---
    st.subheader("📊 Technical Telemetry")
    
    # Render Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resolution", report['metrics']['Resolution'])
    m2.metric("Sharpness (Focus)", report['metrics']['Sharpness'])
    m3.metric("Lighting (Brightness)", report['metrics']['Brightness'])
    m4.metric("File Weight", report['metrics']['File Size'])

    st.write("") # spacing
    
    # Render Dominant Color Palette using Custom HTML
    st.write("**Extracted Product Color Palette:**")
    color_html = ""
    for hex_code in report['colors']:
        # This draws the beautiful colored squares
        color_html += f"""
        <div style="
            background-color: {hex_code}; 
            width: 60px; 
            height: 60px; 
            border-radius: 8px; 
            display: inline-block; 
            margin-right: 15px; 
            border: 1px solid #444;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            " title="{hex_code}">
        </div>
        """
    st.markdown(color_html, unsafe_allow_html=True)
    st.caption(f"Hex Codes: {', '.join(report['colors'])}")

    st.write("") # spacing

    # Provide the Detailed Reason / Assessment
    st.write("**Verbose System Assessment:**")
    for reason in report.get('reasons', []):
        if "REJECTED" in report['status']:
            st.warning(f"⚠️ {reason}")
        else:
            st.success(f"✅ {reason}")

    if os.path.exists(temp_path):
        os.remove(temp_path)
