import cv2
from tracker1 import ObjectCounter  # Importando ObjectCounter desde tracker.py

# Definir la función de callback del mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Verificar el movimiento del mouse
        point = [x, y]
        print(f"El mouse se movió a: {point}")

# Abrir el archivo de video
cap = cv2.VideoCapture('prueba.mp4')

# Definir los puntos de la región para el conteo
# Estos puntos pueden ajustarse dependiendo del área donde quieres contar los vehículos en el video.
region_points = [(10, 10), (1000, 490)]  # Ajusta esta región según el video

# Inicializar el contador de objetos
counter = ObjectCounter(
    region=region_points,  # Pasar los puntos de la región
    model="matricula.pt",  # Modelo para el conteo de objetos (ajustar si es necesario)
    classes=[0],
    show_in=True,  # Mostrar conteo de entrada
    show_out=True,  # Mostrar conteo de salida
    line_width=2,  # Ajustar el grosor de la línea para la visualización
)

# Crear una ventana con nombre y asignar la función de callback del mouse
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

count = 0

while True:
    # Leer un fotograma del video
    ret, frame = cap.read()
    if not ret:
        break
        # Si el video termina, reiniciar desde el principio
#        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#        continue
    count += 1
    if count % 2 != 0:  # Omitir fotogramas impares para optimizar el rendimiento
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Procesar el fotograma con el contador de objetos
    frame = counter.count(frame)

    # Mostrar el fotograma
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Presionar 'q' para salir
        break

# Liberar el objeto de captura de video y cerrar la ventana de visualización
cap.release()
cv2.destroyAllWindows()
