import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

# ======================================================
# 1. Funciones auxiliares
# ======================================================

def edge_density(edge):
    return np.count_nonzero(edge) / edge.size

def edge_entropy(edge):
    hist, _ = np.histogram(edge.flatten(), bins=256, range=(0, 255), density=True)
    return scipy_entropy(hist + 1e-12)

def contour_length(edge):
    edge_uint = np.uint8(edge > 0) * 255
    contours, _ = cv2.findContours(edge_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(len(c) for c in contours)

def add_gaussian_noise(image, sigma=15):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_blur(image, k=5):
    return cv2.GaussianBlur(image, (k, k), 0)


# ======================================================
# 2. Detectores de bordes
# ======================================================

def detect_edges_all(img, gray, net):

    # Canny
    canny = cv2.Canny(gray, 100, 200)

    # Sobel
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.uint8(255 * cv2.magnitude(sx, sy) / np.max(cv2.magnitude(sx, sy)))

    # Laplaciano
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(255 * np.abs(lap) / np.max(np.abs(lap)))

    # Scharr
    schx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    schy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr = np.uint8(255 * cv2.magnitude(schx, schy) / np.max(cv2.magnitude(schx, schy)))

    # DoG
    b1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
    b2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    dog = np.uint8(255 * np.abs(b1 - b2) / np.max(np.abs(b1 - b2)))

    # CED (difusión anisotrópica + Sobel)
    def anisotropic(img, niter=15, kappa=30, gamma=0.1):
        img = img.astype(np.float32)
        for _ in range(niter):
            nablaN = np.roll(img, -1, axis=0) - img
            nablaS = np.roll(img,  1, axis=0) - img
            nablaE = np.roll(img, -1, axis=1) - img
            nablaW = np.roll(img,  1, axis=1) - img

            cN = np.exp(-(nablaN/kappa)**2)
            cS = np.exp(-(nablaS/kappa)**2)
            cE = np.exp(-(nablaE/kappa)**2)
            cW = np.exp(-(nablaW/kappa)**2)

            img += gamma*(cN*nablaN + cS*nablaS + cE*nablaE + cW*nablaW)
        return img

    diff = anisotropic(gray)
    dx = cv2.Sobel(diff, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(diff, cv2.CV_64F, 0, 1)
    ced = np.uint8(255 * cv2.magnitude(dx, dy) / np.max(cv2.magnitude(dx, dy)))

    # HED
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0,
        size=(img.shape[1], img.shape[0]),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    hed_out = net.forward()[0,0]
    hed = np.uint8(255 * hed_out)

    return {
        "Canny": canny,
        "Sobel": sobel,
        "Laplaciano": lap,
        "Scharr": scharr,
        "DoG": dog,
        "CED": ced,
        "HED": hed
    }


# ======================================================
# 3. Cargar imagen y HED
# ======================================================

img = cv2.imread("t083.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

proto = "deploy.prototxt"
model = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)

edges = detect_edges_all(img, gray, net)

# ======================================================
# 4. Calcular métricas
# ======================================================

metrics = {}

for name, e in edges.items():
    dens = edge_density(e)
    ent  = edge_entropy(e)
    cont = contour_length(e)

    # estabilidad
    noisy = add_gaussian_noise(gray)
    blur  = add_blur(gray)

    e_noisy = detect_edges_all(img, noisy, net)[name]
    e_blur  = detect_edges_all(img, blur,  net)[name]

    # diferencia promedio con ruido + blur
    stability = 1.0 - (
        np.mean(np.abs(e - e_noisy))/255 +
        np.mean(np.abs(e - e_blur))/255
    )/2

    metrics[name] = {
        "Densidad": dens,
        "Entropía": ent,
        "Contornos": cont,
        "Estabilidad": stability
    }

# ======================================================
# 5. Mostrar ranking
# ======================================================

print("\n=== Métricas por detector ===\n")
for k,v in metrics.items():
    print(f"{k}: {v}")

print("\n=== Ranking final ===\n")

# criterios: estabilidad + contornos largos - entropía excesiva
ranking = sorted(
    metrics.items(),
    key=lambda x: (
        + x[1]["Estabilidad"],
        + x[1]["Contornos"],
        - x[1]["Entropía"]
    ),
    reverse=True
)

for i,(name,vals) in enumerate(ranking,1):
    print(f"{i}. {name}  ->  Score {vals}")
