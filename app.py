import streamlit as st
import base64
from ml import MLModel
from dl import DLModel

st.set_page_config(page_title="Drawing with LLM", page_icon="🎨", layout="wide")

@st.cache_resource
def load_ml_model():
    return MLModel(device="cuda" if st.session_state.get("use_gpu", True) else "cpu")

@st.cache_resource
def load_dl_model():
    return DLModel(device="cuda" if st.session_state.get("use_gpu", True) else "cpu")

def render_svg(svg_content):
    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}" width="100%" height="auto"/>'

st.title("Drawing with LLM 🎨")

with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model Type", ["ML Model (vtracer)", "DL Model (starvector)"])
    use_gpu = st.checkbox("Use GPU", value=True)
    st.session_state["use_gpu"] = use_gpu
    
    if model_type == "ML Model (vtracer)":
        st.subheader("ML Model Settings")
        simplify = st.checkbox("Simplify SVG", value=True)
        color_precision = st.slider("Color Precision", 1, 10, 6)
        filter_speckle = st.slider("Filter Speckle", 0, 10, 4)
        path_precision = st.slider("Path Precision", 1, 10, 8)

prompt = st.text_area("Enter your description", "A cat sitting on a windowsill at sunset")

if st.button("Generate SVG"):
    with st.spinner("Generating SVG..."):
        if model_type == "ML Model (vtracer)":
            model = load_ml_model()
            svg_content = model.predict(
                prompt,
                simplify=simplify,
                color_precision=color_precision,
                filter_speckle=filter_speckle,
                path_precision=path_precision
            )
        else:
            model = load_dl_model()
            svg_content = model.predict(prompt)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generated SVG")
            st.markdown(render_svg(svg_content), unsafe_allow_html=True)
        
        with col2:
            st.subheader("SVG Code")
            st.code(svg_content, language="xml")
            
            # Download button for SVG
            st.download_button(
                label="Download SVG",
                data=svg_content,
                file_name="generated_svg.svg",
                mime="image/svg+xml"
            )

st.markdown("---")
st.markdown("This app uses Stable Diffusion to generate images from text and converts them to SVG.")
