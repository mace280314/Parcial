from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt')  

# Directorio donde se guardarán las imágenes procesadas
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Función para leer la imagen
def read_imagefile(file) -> np.ndarray:
    image = np.array(Image.open(io.BytesIO(file)))
    return image

@app.get("/")  # Ruta para mostrar el formulario de subida de archivos
async def main():
    content = """
    <html>
    <body>
        <h1>Sube una imagen para clasificar</h1>
        <form action="/classify" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Subir Imagen">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Leer la imagen cargada
    image_data = await file.read()
    image = read_imagefile(image_data)

    # Ejecutar detección de objetos con YOLOv8
    results = model(image)

    # Dibujar las cajas delimitadoras y etiquetas en la imagen
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
        confs = result.boxes.conf.cpu().numpy()  # Confianza
        clss = result.boxes.cls.cpu().numpy()    # Clases
        for box, conf, cls in zip(boxes, confs, clss):
            label = f"{model.names[int(cls)]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Guardar la imagen procesada
    output_path = os.path.join(output_dir, f"processed_{file.filename}")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Generar el HTML para mostrar la imagen original y la procesada, junto con las detecciones
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                background-color: #111;
                color: white;
                font-family: Arial, sans-serif;
            }}
            h1 {{
                color: green;
                text-align: center;
            }}
            .container {{
                display: flex;
                justify-content: space-around;
                padding: 20px;
            }}
            .image-container {{
                max-width: 45%;
            }}
            .details {{
                max-width: 45%;
            }}
            img {{
                width: 100%;
                height: auto;
                border: 2px solid white;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th {{
                background-color: green;
                color: white;
                padding: 10px;
            }}
            td {{
                padding: 10px;
                text-align: center;
                border: 1px solid white;
            }}
        </style>
    </head>
    <body>
        <h1>Objetos Detectados</h1>
        <div class="container">
            <div class="image-container">
                <h2>Imagen Original</h2>
                <img src="/output/{file.filename}" alt="Imagen original">
                <h2>Imagen Procesada</h2>
                <img src="/output/processed_{file.filename}" alt="Imagen procesada">
            </div>
            <div class="details">
                <h2>Detalles de la Detección</h2>
                <table>
                    <tr>
                        <th>Clase</th>
                        <th>Confianza</th>
                        <th>Coordenadas [x1, y1, x2, y2]</th>
                    </tr>
    """

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confs, clss):
            html_content += f"""
            <tr>
                <td>{model.names[int(cls)]}</td>
                <td>{conf:.2f}</td>
                <td>{[round(b, 2) for b in box]}</td>
            </tr>
            """

    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

@app.get("/output/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(output_dir, filename)
    return FileResponse(file_path)
