import numpy as np
import matplotlib.pyplot as plt

# Crear una imagen en blanco y negro (grayscale)
img = np.linspace(0, 255, 256).astype(np.uint8)
img = np.tile(img, (256, 1))  # 256x256

# --- Funciones de ruido ---

def ruido_gaussiano(img, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def ruido_sal_pimienta(img, prob=0.02):
    noisy = img.copy()
    num_pixels = img.size
    # Sal
    coords = np.random.choice(num_pixels, int(num_pixels * prob / 2), replace=False)
    noisy.reshape(-1)[coords] = 255
    # Pimienta
    coords = np.random.choice(num_pixels, int(num_pixels * prob / 2), replace=False)
    noisy.reshape(-1)[coords] = 0
    return noisy

def ruido_speckle(img, sigma=0.1):
    gauss = np.random.randn(*img.shape)
    noisy = img + img * gauss * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Generar ruidos
gauss = ruido_gaussiano(img)
salpim = ruido_sal_pimienta(img)
speckle = ruido_speckle(img)

# Mostrar imágenes
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.title("Original (B/N)")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Ruido Gaussiano")
plt.imshow(gauss, cmap="gray")
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Sal y Pimienta")
plt.imshow(salpim, cmap="gray")
plt.axis("off")

plt.subplot(2,2,4)
plt.title("Speckle")
plt.imshow(speckle, cmap="gray")
plt.axis("off")

plt.show()
