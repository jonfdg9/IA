import cv2
from tracker1 import ObjectCounter  # Importando ObjectCounter desde tracker.py

# Definir la función de callback del mouse
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Verificar el movimiento del mouse
        point = [x, y]
        print(f"El mouse se movió a: {point}")

# Abrir el archivo de video
cap = cv2.VideoCapture('trafico.mp4')

# Verificar si el video se ha abierto correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Definir los puntos de la región para el conteo
region_points = [(365, 248), (750, 248)]  # Ajusta esta región según el video

# Inicializar el contador de objetos
counter = ObjectCounter(
    region=region_points,  # Pasar los puntos de la región
    model="yolo11s.pt",  # Modelo para el conteo de objetos (ajustar si es necesario)
    classes=[2],  # Detectar solo "car" (clase 2)
    show_in=True,  # Mostrar conteo de entrada
    show_out=True,  # Mostrar conteo de salida
    line_width=2,  # Ajustar el grosor de la línea para la visualización
)

# Crear una ventana con nombre y asignar la función de callback del mouse
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

count = 0

# Obtener las propiedades del video (ancho, alto y FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Ajustar el tamaño del fotograma (si es necesario) para coincidir con el tamaño original
frame_size = (frame_width, frame_height)

# Crear el objeto VideoWriter para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usar el códec 'mp4v' para .mp4
out = cv2.VideoWriter('resultado.mp4', fourcc, fps, frame_size)

if not out.isOpened():
    print("Error al abrir el archivo de salida.")
    exit()

while True:
    # Leer un fotograma del video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:  # Omitir fotogramas impares para optimizar el rendimiento
        continue

    # Procesar el fotograma con el contador de objetos
    frame = counter.count(frame)

    # Escribir el fotograma procesado en el archivo de salida
    out.write(frame)

    # Mostrar el fotograma procesado
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Presionar 'q' para salir
        break

# Liberar los recursos
cap.release()
out.release()  # Liberar el VideoWriter
cv2.destroyAllWindows()

# Imprimir el conteo final por terminal
print(f"Conteo final de coches entrantes: {counter.in_count}")
print(f"Conteo final de coches salientes: {counter.out_count}")