import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.serialization
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#########################################
# 1. Definición del modelo DnCNN (igual al original)
#########################################
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

#########################################
# 2. Cargar modelo
#########################################
torch.serialization.add_safe_globals([DnCNN])
model = DnCNN(channels=1)
loaded = torch.load("model.pth", map_location="cpu", weights_only=False)
model = loaded  # ya es el modelo completo
model.eval()

#########################################
# 3. Cargar imagen
#########################################
img = Image.open("couple.tif").convert("L")  # escala de grises
transform = T.ToTensor()
img_tensor = transform(img)

#########################################
# 4. Agregar ruido gaussiano
#########################################
noise_level = 25 / 255.0
noise = torch.randn(img_tensor.size()) * noise_level
noisy_img_tensor = img_tensor + noise
noisy_img_tensor = torch.clamp(noisy_img_tensor, 0., 1.)

#########################################
# 5. Pasar por la red
#########################################
with torch.no_grad():
    input_batch = noisy_img_tensor.unsqueeze(0)  # (1,1,H,W)
    denoised = model(input_batch).squeeze(0)

#########################################
# 6. Guardar resultados
#########################################
to_pil = T.ToPILImage()

to_pil(noisy_img_tensor).save("imagen_con_ruido.png")
to_pil(denoised).save("imagen_denoised.png")

print("Listo! Se guardaron:")
print("- imagen_con_ruido.png")
print("- imagen_denoised.png")


#########################################
# 7. Mostrar imágenes para comparar
#########################################

plt.figure(figsize=(12, 4))

# Imagen original
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

# Imagen con ruido
plt.subplot(1, 3, 2)
plt.title("Con ruido")
plt.imshow(noisy_img_tensor.squeeze().numpy(), cmap="gray")
plt.axis("off")

# Imagen denoised
plt.subplot(1, 3, 3)
plt.title("Denoised (DnCNN)")
plt.imshow(denoised.squeeze().numpy(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


# Convertir tensores a numpy
original_np = img_tensor.squeeze().numpy()
noisy_np = noisy_img_tensor.squeeze().numpy()
denoised_np = denoised.squeeze().numpy()

# PSNR
psnr_noisy = peak_signal_noise_ratio(original_np, noisy_np, data_range=1.0)
psnr_denoised = peak_signal_noise_ratio(original_np, denoised_np, data_range=1.0)

# SSIM
ssim_noisy = structural_similarity(original_np, noisy_np, data_range=1.0)
ssim_denoised = structural_similarity(original_np, denoised_np, data_range=1.0)

print("\n### COMPARACIÓN NUMÉRICA ###")
print(f"PSNR imagen ruidosa    : {psnr_noisy:.2f} dB")
print(f"PSNR imagen denoised   : {psnr_denoised:.2f} dB")
print(f"SSIM imagen ruidosa    : {ssim_noisy:.4f}")
print(f"SSIM imagen denoised   : {ssim_denoised:.4f}")