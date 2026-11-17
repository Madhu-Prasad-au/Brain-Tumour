import streamlit as st
import torch
from PIL import Image
from brain_tumor import load_for_inference,predict_image

st.set_page_config(page_title="Brain MRI Classifier",page_icon="🧠",layout="centered")
st.title("Brain Tumor MRI Classification")
st.caption("Research and educational image classification demo")
st.warning("This AI output is not a medical diagnosis. MRI images must be reviewed by a qualified radiologist.")
@st.cache_resource
def get_model(path): return load_for_inference(path,device='cpu')
model_file=st.sidebar.selectbox("Model checkpoint",["models/cnn.pt","models/mobilenet_v2.pt"])
upload=st.file_uploader("Upload an MRI image",type=["jpg","jpeg","png","bmp"])
if upload:
    image=Image.open(upload).convert('RGB')
    st.image(image,caption="Uploaded MRI",use_container_width=True)
    try:
        model,meta=get_model(model_file)
        scores=predict_image(model,image,meta['image_size'])
        name,confidence=scores[0]
        st.subheader(name.replace('_',' ').title())
        st.metric("Model confidence",f"{confidence:.1%}")
        st.write("Class scores")
        st.bar_chart({n.replace('_',' ').title():v for n,v in scores})
        st.caption(f"Model: {meta['model']} · input size: {meta['image_size']}×{meta['image_size']}")
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {model_file}. Train the corresponding notebook first.")
    except Exception as e:
        st.error(f"Could not classify this image: {e}")
