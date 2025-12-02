import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. Cargar imagen RGB (3 canales)
# ======================================================
img = cv2.imread("t083.png")
if img is None:
    raise FileNotFoundError("No se encontró t083.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ======================================================
# 2. Filtros clásicos
# ======================================================

# Canny
edges_canny = cv2.Canny(gray, 100, 200)

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobelx, sobely)
edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))

# Laplaciano
edges_laplace = cv2.Laplacian(gray, cv2.CV_64F)
edges_laplace = np.uint8(255 * np.abs(edges_laplace) / np.max(np.abs(edges_laplace)))

# Scharr
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
edges_scharr = cv2.magnitude(scharrx, scharry)
edges_scharr = np.uint8(255 * edges_scharr / np.max(edges_scharr))

# DoG (Difference of Gaussians)
blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
edges_dog = cv2.absdiff(blur1, blur2)
edges_dog = np.uint8(255 * edges_dog / np.max(edges_dog))

# ======================================================
# 3. Aproximación CED (Anisotropic diffusion + Sobel)
# ======================================================
def anisotropic_diffusion(img, niter=15, kappa=30, gamma=0.1):
    img = img.astype("float32")
    for i in range(niter):
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img, 1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img, 1, axis=1) - img

        cN = np.exp(-(nablaN / kappa)**2)
        cS = np.exp(-(nablaS / kappa)**2)
        cE = np.exp(-(nablaE / kappa)**2)
        cW = np.exp(-(nablaW / kappa)**2)

        img += gamma * (
            cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW
        )
    return img

diffused = anisotropic_diffusion(gray, niter=15)
sobelx_d = cv2.Sobel(diffused, cv2.CV_64F, 1, 0)
sobely_d = cv2.Sobel(diffused, cv2.CV_64F, 0, 1)
edges_ced = cv2.magnitude(sobelx_d, sobely_d)
edges_ced = np.uint8(255 * edges_ced / np.max(edges_ced))

# ======================================================
# 4. HED Deep Learning
# ======================================================
proto = "deploy.prototxt"
model = "hed_pretrained_bsds.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, model)

blob = cv2.dnn.blobFromImage(
    img, scalefactor=1.0,
    size=(img.shape[1], img.shape[0]),
    mean=(104.00698793, 116.66876762, 122.67891434),
    swapRB=False, crop=False
)

net.setInput(blob)
hed = net.forward()
hed = hed[0, 0]
hed = cv2.resize(hed, (img.shape[1], img.shape[0]))
edges_hed = np.uint8(255 * hed)

# ======================================================
# 5. Mostrar todos los resultados
# ======================================================
titles = [
    "Original RGB", "Canny", "Sobel", "Laplaciano", 
    "Scharr", "DoG", "CED (anisotropic)", "HED (Deep)"
]

images = [
    img_rgb, edges_canny, edges_sobel, edges_laplace,
    edges_scharr, edges_dog, edges_ced, edges_hed
]

plt.figure(figsize=(18, 12))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
