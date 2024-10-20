from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import transforms
import torch
from PIL import Image
import io
import torchvision

app = FastAPI()

# Cargar el modelo ResNet18 entrenado en CIFAR-100, forzando el uso de la CPU
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(torch.load('cifar100_model.pkl', map_location=torch.device('cpu')))
model.eval()

# Lista completa de las 100 clases de CIFAR-100
cifar100_classes = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Transformación para las imágenes de CIFAR-100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Página inicial para subir la imagen
@app.get("/", response_class=HTMLResponse)
async def upload_form():
    html_content = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #121212;
                color: #fff;
            }
            .container {
                width: 60%;
                padding: 20px;
                border-radius: 10px;
                background-color: #1f1f1f;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            }
            h1 {
                text-align: center;
                color: #00ff00;
            }
            .form-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            input[type="file"] {
                background-color: #222;
                padding: 10px;
                color: #fff;
                border-radius: 5px;
                border: none;
                cursor: pointer;
            }
            input[type="submit"] {
                background-color: #00ff00;
                color: #000;
                padding: 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }
            input[type="submit"]:hover {
                background-color: #00cc00;
            }
            .image-container img {
                max-width: 100%;
                border-radius: 10px;
            }
            .result {
                color: #00ff00;
                font-weight: bold;
                text-align: center;
                margin-top: 20px;
            }
            .error {
                color: #ff0000;
                text-align: center;
                margin-top: 20px;
            }
            .classification-result {
                text-align: center;
                margin-top: 20px;
                font-size: 18px;
            }
            .classification-result strong {
                color: #00ff00;
                font-size: 20px;
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clasificación de Imágenes con CIFAR-100</h1>
            <form action="/classify" enctype="multipart/form-data" method="post">
                <div class="form-container">
                    <input name="file" type="file" accept="image/*">
                    <input type="submit" value="Clasificar">
                </div>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint para clasificar las imágenes
@app.post("/classify", response_class=HTMLResponse)
async def classify_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida
        image = Image.open(io.BytesIO(await file.read()))
        # Preprocesar la imagen
        image_tensor = transform(image).unsqueeze(0)
        # Mover el tensor a la CPU
        image_tensor = image_tensor.to(torch.device('cpu'))

        # Hacer la predicción en la CPU
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class_idx = predicted.item()

            # Verificar si el índice está dentro de los límites
            if predicted_class_idx >= 0 and predicted_class_idx < len(cifar100_classes):
                class_name = cifar100_classes[predicted_class_idx]
            else:
                class_name = "Clasificación fuera de rango"

        # Convertir la imagen a base64 para mostrarla en HTML
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # HTML con los resultados
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #121212;
                    color: #fff;
                }}
                .container {{
                    width: 60%;
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #1f1f1f;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                }}
                .image-container img {{
                    max-width: 100%;
                    border-radius: 10px;
                }}
                .result {{
                    color: #00ff00;
                    font-weight: bold;
                    text-align: center;
                    margin-top: 20px;
                }}
                h1 {{
                    text-align: center;
                    color: #00ff00;
                }}
                .classification-result {{
                    text-align: center;
                    margin-top: 20px;
                    font-size: 18px;
                }}
                .classification-result strong {{
                    color: #00ff00;
                    font-size: 20px;
                    text-decoration: underline;
                }}
                p {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Clasificación de la Imagen</h1>
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{img_str}" alt="Imagen cargada">
                </div>
                <div class="classification-result">
                    <p>Clasificación: <strong>{class_name}</strong></p>
                </div>
                <p><a href="/">Subir otra imagen</a></p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(content=f"<h1 class='error'>Error: {str(e)}</h1>", status_code=500)
