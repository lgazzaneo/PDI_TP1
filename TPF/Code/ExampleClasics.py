import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from PIL import Image
import Noises

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

# 1. Convertir de float32-normalizada a uint8
img_uint8 = (img_np * 255).astype(np.uint8)

# 2. Aplicar ruido sin cambiar tus funciones
sp_uint8 = Noises.ruido_sal_pimienta(img_uint8)
speckle_uint8 = Noises.ruido_speckle(img_uint8)

# 3. Volver a normalizar si lo querés para DnCNN o display
sp_img = sp_uint8.astype(np.float32) / 255.0
speckle_img = speckle_uint8.astype(np.float32) / 255.0


#########################################
# 3. Filtros tradicionales
#########################################

# --- PASABAJOS (Gaussiano) ---
lowpass = cv2.GaussianBlur(noisy_img, (5, 5), 1)
lowpass_sp = cv2.GaussianBlur(sp_img, (5, 5), 1)
lowpass_speckle = cv2.GaussianBlur(speckle_img, (5, 5), 1)

# --- PASAALTOS (Laplaciano) ---
# Asegurar float32 compatible con OpenCV
noisy32 = noisy_img.astype(np.float32)
noisy32_sp = sp_img.astype(np.float32)
noisy32_speckle = speckle_img.astype(np.float32)
# Laplaciano
lap = cv2.Laplacian(noisy32, cv2.CV_32F)
lap_sp = cv2.Laplacian(noisy32_sp, cv2.CV_32F)
lap_speckle = cv2.Laplacian(noisy32_speckle, cv2.CV_32F)
# Imagen pasaaltos
highpass = np.clip(noisy32 + lap, 0, 1)
highpass_sp = np.clip(noisy32_sp + lap_sp, 0, 1)
highpass_speckle = np.clip(noisy32_speckle + lap_speckle, 0, 1)

# --- MEDIANA ---
median = cv2.medianBlur((noisy_img * 255).astype(np.uint8), 3)
median = median.astype(np.float32) / 255.0
median_sp = cv2.medianBlur((sp_img * 255).astype(np.uint8), 3)
median_sp = median_sp.astype(np.float32) / 255.0
median_speckle = cv2.medianBlur((speckle_img * 255).astype(np.uint8), 3)
median_speckle = median_speckle.astype(np.float32) / 255.0

#########################################
# 4. Comparación numérica
#########################################
psnr_low = peak_signal_noise_ratio(img_np, lowpass, data_range=1.0)
psnr_high = peak_signal_noise_ratio(img_np, highpass, data_range=1.0)
psnr_med = peak_signal_noise_ratio(img_np, median, data_range=1.0)

psnr_low_sp = peak_signal_noise_ratio(img_np, lowpass_sp, data_range=1.0)
psnr_high_sp = peak_signal_noise_ratio(img_np, highpass_sp, data_range=1.0)
psnr_med_sp = peak_signal_noise_ratio(img_np, median_sp, data_range=1.0)

psnr_low_speckle = peak_signal_noise_ratio(img_np, lowpass_speckle, data_range=1.0)
psnr_high_speckle = peak_signal_noise_ratio(img_np, highpass_speckle, data_range=1.0)
psnr_med_speckle = peak_signal_noise_ratio(img_np, median_speckle, data_range=1.0)

ssim_low = structural_similarity(img_np, lowpass, data_range=1.0)
ssim_high = structural_similarity(img_np, highpass, data_range=1.0)
ssim_med = structural_similarity(img_np, median, data_range=1.0)

ssim_low_sp = structural_similarity(img_np, lowpass_sp, data_range=1.0)
ssim_high_sp = structural_similarity(img_np, highpass_sp, data_range=1.0)
ssim_med_sp = structural_similarity(img_np, median_sp, data_range=1.0)

ssim_low_speckle = structural_similarity(img_np, lowpass_speckle, data_range=1.0)
ssim_high_speckle = structural_similarity(img_np, highpass_speckle, data_range=1.0)
ssim_med_speckle = structural_similarity(img_np, median_speckle, data_range=1.0)

print("\n### RESULTADOS NUMÉRICOS ###")
print(f"PSNR Lowpass (Gauss)   : {psnr_low:.2f} dB")
print(f"PSNR Highpass (Lap)    : {psnr_high:.2f} dB")
print(f"PSNR Mediana           : {psnr_med:.2f} dB")
print()
print(f"SSIM Lowpass           : {ssim_low:.4f}")
print(f"SSIM Highpass          : {ssim_high:.4f}")
print(f"SSIM Mediana           : {ssim_med:.4f}")
print()
print(f"PSNR Lowpass (Sal&Pim) : {psnr_low_sp:.2f} dB")
print(f"PSNR Highpass (Sal&Pim): {psnr_high_sp:.2f} dB")
print(f"PSNR Mediana (Sal&Pim) : {psnr_med_sp:.2f} dB")
print()
print(f"SSIM Lowpass (Sal&Pim) : {ssim_low_sp:.4f}")
print(f"SSIM Highpass (Sal&Pim): {ssim_high_sp:.4f}")
print(f"SSIM Mediana (Sal&Pim) : {ssim_med_sp:.4f}")
print()
print(f"PSNR Lowpass (Speckle) : {psnr_low_speckle:.2f} dB")
print(f"PSNR Highpass (Speckle): {psnr_high_speckle:.2f} dB")
print(f"PSNR Mediana (Speckle) : {psnr_med_speckle:.2f} dB")
print()
print(f"SSIM Lowpass (Speckle) : {ssim_low_speckle:.4f}")
print(f"SSIM Highpass (Speckle): {ssim_high_speckle:.4f}")
print(f"SSIM Mediana (Speckle) : {ssim_med_speckle:.4f}")
print()

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

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_np, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Con ruido Sal y Pimienta")
plt.imshow(sp_img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Pasabajos (Gauss) Sal y Pimienta")
plt.imshow(lowpass_sp, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Pasaaltos (Laplace) Sal y Pimienta")
plt.imshow(highpass_sp, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Mediana Sal y Pimienta")
plt.imshow(median_sp, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_np, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Con ruido Speckle")
plt.imshow(speckle_img, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Pasabajos (Gauss) Speckle")
plt.imshow(lowpass_speckle, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Pasaaltos (Laplace) Speckle")
plt.imshow(highpass_speckle, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Mediana Speckle")
plt.imshow(median_speckle, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
