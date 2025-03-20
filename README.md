# jfca-video-recognition
reconocimiento de objetos por la webcam

**# Streamlit Object Detection**

Este proyecto utiliza **Streamlit** y **YOLOv5** para la detección de objetos en tiempo real desde la cámara web.

## **Requisitos**

Asegúrate de tener **Python 3.8+** instalado y ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install streamlit opencv-python ultralytics torch numpy
```

## **Uso**

Ejecuta la aplicación con:

```bash
streamlit run app.py
```

Luego, activa la cámara desde la interfaz para comenzar la detección de objetos en tiempo real.

## **Estructura del Proyecto**

```
project/
│── app.py
│── README.md
│── .gitignore
```

## **.gitignore**

Este archivo evita subir archivos innecesarios al repositorio:

```
__pycache__/
*.pyc
.DS_Store
venv/
.env
