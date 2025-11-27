import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from PIL import Image

#########################################
# 1. Cargar imagen
#########################################
img = Image.open("couple.tif").convert("L")      # en escala de grises
img_np = np.array(img).astype(np.float32) / 255.0   # normalizar 0-1

#########################################
# 2. Agregar ruido gaussiano
#########################################
noise_sigma = 25 / 255.0  # mismo nivel que DnCNN sigma=25
noise = np.random.normal(0, noise_sigma, img_np.shape)
noisy_img = np.clip(img_np + noise, 0, 1)

#########################################
# 3. Filtros tradicionales
#########################################

# --- PASABAJOS (Gaussiano) ---
lowpass = cv2.GaussianBlur(noisy_img, (5, 5), 1)

# --- PASAALTOS (Laplaciano) ---
# Asegurar float32 compatible con OpenCV
noisy32 = noisy_img.astype(np.float32)
# Laplaciano
lap = cv2.Laplacian(noisy32, cv2.CV_32F)
# Imagen pasaaltos
highpass = np.clip(noisy32 + lap, 0, 1)

# --- MEDIANA ---
median = cv2.medianBlur((noisy_img * 255).astype(np.uint8), 3)
median = median.astype(np.float32) / 255.0

#########################################
# 4. Comparación numérica
#########################################
psnr_low = peak_signal_noise_ratio(img_np, lowpass, data_range=1.0)
psnr_high = peak_signal_noise_ratio(img_np, highpass, data_range=1.0)
psnr_med = peak_signal_noise_ratio(img_np, median, data_range=1.0)

ssim_low = structural_similarity(img_np, lowpass, data_range=1.0)
ssim_high = structural_similarity(img_np, highpass, data_range=1.0)
ssim_med = structural_similarity(img_np, median, data_range=1.0)

print("\n### RESULTADOS NUMÉRICOS ###")
print(f"PSNR Lowpass (Gauss)   : {psnr_low:.2f} dB")
print(f"PSNR Highpass (Lap)    : {psnr_high:.2f} dB")
print(f"PSNR Mediana           : {psnr_med:.2f} dB")
print()
print(f"SSIM Lowpass           : {ssim_low:.4f}")
print(f"SSIM Highpass          : {ssim_high:.4f}")
print(f"SSIM Mediana           : {ssim_med:.4f}")

#########################################
# 5. Mostrar imágenes
#########################################
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_np, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Con ruido")
plt.imshow(noisy_img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Pasabajos (Gauss)")
plt.imshow(lowpass, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Pasaaltos (Laplace)")
plt.imshow(highpass, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Mediana")
plt.imshow(median, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
