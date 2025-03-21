import streamlit as st
import cv2
import tempfile
import torch
from ultralytics import YOLO
import numpy as np

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a "yolov8s.pt" para un modelo más preciso

# Configuración de la aplicación
st.title("Detección de Objetos en Video con YOLOv8")
st.write("Sube un video y el modelo detectará los objetos en él.")

# Subir video
video_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    # Guardar el archivo temporalmente
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    # Abrir el video
    cap = cv2.VideoCapture(tfile.name)
    
    # Obtener FPS y tamaño del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Crear espacio para el video procesado
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir frame a RGB (Streamlit usa RGB, OpenCV usa BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar objetos
        results = model(frame_rgb)
        
        # Dibujar los resultados en el frame
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"{model.names[int(cls)]} ({conf:.2f})"  # Nombre del objeto con confianza
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar el frame con detecciones en Streamlit
        stframe.image(frame_rgb, channels="RGB")
        
    cap.release()
    st.success("Procesamiento finalizado.")