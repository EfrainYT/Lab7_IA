
from ultralytics import YOLO
import cv2

# ======================
# CONFIGURACIÓN
# ======================

# Ruta al modelo entrenado
MODEL_PATH = r'C:\Users\Usuario\Desktop\Lab7_IA\runs\detect\custom_yolo_model2\weights\best.pt'

# Cargar el modelo
print("Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)
print("Modelo cargado correctamente.")

# Umbral de confianza
CONF_THRES = 0.014

# ======================
# CAPTURA DE VIDEO
# ======================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(" No se pudo acceder a la cámara.")
    exit()

print("🎥 Cámara iniciada. Presiona 'q' para salir.")

# ======================
# BUCLE PRINCIPAL
# ======================

while True:
    ret, frame = cap.read()
    if not ret:
        print(" No se pudo leer frame de la cámara.")
        break

    # Inferencia con YOLO
    results = model(frame, conf=CONF_THRES)

    # Dibujar resultados sobre el frame
    annotated_frame = results[0].plot()

    # Mostrar en ventana
    cv2.imshow("Detección YOLO - Webcam", annotated_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# FINALIZAR
# ======================

cap.release()
cv2.destroyAllWindows()
print("Cámara cerrada correctamente.")
