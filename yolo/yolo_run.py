from ultralytics import YOLO
import torch

print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

model = YOLO('') # Colocamos el modelo a entrenar (por ejemplo, 'yolov8n.pt' o 'yolov8s.pt')

# Entrenar el modelo
results = model.train(
    data='', # Añadimos la ruta del archivo de datos .YAML del conjunto personalizado (dataset)
    epochs=150,                       # Número de épocas
    imgsz=640,                        # Tamaño de imagen
    batch=16,                         # Batch size (ajusta según tu VRAM)
    device=0,                         # GPU 0 (o 'cuda')
    workers=8,                        # Workers para cargar datos
    patience=50,                      # Early stopping
    save=True,                        # Guardar checkpoints
    project='runs/train',             # Carpeta de salida
    name=''             # Nombre del experimento
)