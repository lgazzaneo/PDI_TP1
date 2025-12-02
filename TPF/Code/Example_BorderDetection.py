import cv2
import matplotlib.pyplot as plt

# -------------------------
# 1. Cargar imagen (3 canales)
# -------------------------
img = cv2.imread("t083.png")         # <-- color por defecto
if img is None:
    raise FileNotFoundError("No se encontró t083.png")

# Convertimos a RGB solo para mostrar en matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Para Canny usamos escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges_canny = cv2.Canny(gray, 100, 200)

# -------------------------
# 2. Cargar HED
# -------------------------
proto = "deploy.prototxt"
model = "hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, model)

# -------------------------
# 3. Crear blob (IMPORTANTE: siempre 3 canales)
# -------------------------
blob = cv2.dnn.blobFromImage(
    img,                                # imagen BGR 3 canales
    scalefactor=1.0,
    size=(img.shape[1], img.shape[0]),
    mean=(104.00698793, 116.66876762, 122.67891434),
    swapRB=False,
    crop=False
)

net.setInput(blob)
hed = net.forward()

# -------------------------
# 4. Procesar salida HED
# -------------------------
hed = hed[0, 0]
hed = cv2.resize(hed, (img.shape[1], img.shape[0]))
hed = (255 * hed).astype("uint8")

# -------------------------
# 5. Mostrar resultados
# -------------------------
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title("Imagen Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Canny")
plt.imshow(edges_canny, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("HED")
plt.imshow(hed, cmap="gray")
plt.axis("off")

plt.show()
