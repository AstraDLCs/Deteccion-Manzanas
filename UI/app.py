import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

best_weights = "C:/Users/nicko/Downloads/manzanas/runs/detect/train/weights/best.pt"
model = YOLO(best_weights)

st.set_page_config(layout="wide")
st.title("Deteccion de estados en las manzanas")

imagen = st.camera_input("Toma una foto")

if imagen:

    img_pil = Image.open(imagen)
    img_np = np.array(img_pil)
    results = model(img_np)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Imagen con Detecciones")