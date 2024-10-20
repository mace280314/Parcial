# Usar una imagen base de FastAPI con Python
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos al contenedor
COPY requirements.txt .

# Instalar las dependencias en el contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos al contenedor
COPY . .

# Exponer el puerto 8000 para FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
