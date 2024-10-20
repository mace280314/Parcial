import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Cargar el modelo de generación de imágenes usando diffusers
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
image_generator = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Modelo de clasificación de imágenes
image_classifier = pipeline('image-classification', model='microsoft/resnet-50')

st.title("Aplicación de Generación y Clasificación de Imágenes")

# Columna para la generación de imágenes
col1, col2 = st.columns(2)

with col1:
    st.header("Generación de Imágenes")
    prompt = st.text_input("Ingresa un texto para generar una imagen:")
    
    if prompt:
        st.write("Generando imagen...")
        # Generar imagen a partir del prompt
        with torch.no_grad():
            generated_image = image_generator(prompt).images[0]
        st.image(generated_image, caption="Imagen generada", use_column_width=True)

# Columna para la clasificación de imágenes
with col2:
    st.header("Clasificación de Imágenes")
    uploaded_file = st.file_uploader("Sube una imagen para clasificarla", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Abrir la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Clasificar la imagen
        st.write("Clasificando imagen...")
        results = image_classifier(image)
        
        st.write("Resultados de la clasificación:")
        for result in results:
            st.write(f"- **{result['label']}**: {result['score']*100:.2f}%")

# Clasificación de la imagen generada
if prompt and generated_image is not None:
    st.write("Clasificando la imagen generada...")
    classification_results = image_classifier(generated_image)
    
    for result in classification_results:
        st.write(f"- **{result['label']}**: {result['score']*100:.2f}%")
