import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.serialization
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import Noises

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
# FUNCIONES AUXILIARES
#########################################

def tensor_to_np(t):
    return t.squeeze().numpy()

def show_comparison(title, original, noisy, denoised):
    plt.figure(figsize=(12,4))
    plt.suptitle(title)

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Ruidosa")
    plt.imshow(noisy, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Denoised DnCNN")
    plt.imshow(denoised, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def compute_metrics(original, noisy, denoised):
    psnr_noisy = peak_signal_noise_ratio(original, noisy, data_range=1.0)
    psnr_denoised = peak_signal_noise_ratio(original, denoised, data_range=1.0)

    ssim_noisy = structural_similarity(original, noisy, data_range=1.0)
    ssim_denoised = structural_similarity(original, denoised, data_range=1.0)

    print(f"PSNR (ruidosa)  : {psnr_noisy:.2f} dB")
    print(f"PSNR (denoised) : {psnr_denoised:.2f} dB")
    print(f"SSIM (ruidosa)  : {ssim_noisy:.4f}")
    print(f"SSIM (denoised) : {ssim_denoised:.4f}")
    print("--------------------------------------")

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
# 4. PROCESAR TODOS LOS RUIDOS
#########################################

original_np = tensor_to_np(img_tensor)

# =======================================================
# 4.1 Ruido Gaussiano
# =======================================================
noise_level = 25/255.0
noise_gauss = torch.randn(img_tensor.size()) * noise_level
gauss_tensor = torch.clamp(img_tensor + noise_gauss, 0., 1.)

with torch.no_grad():
    den_gauss = model(gauss_tensor.unsqueeze(0)).squeeze()

show_comparison(
    "Ruido Gaussiano",
    original_np,
    tensor_to_np(gauss_tensor),
    tensor_to_np(den_gauss)
)

compute_metrics(original_np, tensor_to_np(gauss_tensor), tensor_to_np(den_gauss))


# =======================================================
# 4.2 Ruido SAL Y PIMIENTA
# =======================================================

img_uint8 = (original_np * 255).astype(np.uint8)
sp_uint8 = Noises.ruido_sal_pimienta(img_uint8, prob=0.02)
sp_tensor = torch.tensor(sp_uint8 / 255.0, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    den_sp = model(sp_tensor.unsqueeze(0)).squeeze()

show_comparison(
    "Ruido Sal&Pimienta",
    original_np,
    tensor_to_np(sp_tensor),
    tensor_to_np(den_sp)
)

compute_metrics(original_np, tensor_to_np(sp_tensor), tensor_to_np(den_sp))


# =======================================================
# 4.3 Ruido SPECKLE
# =======================================================

speckle_uint8 = Noises.ruido_speckle(img_uint8)
speckle_tensor = torch.tensor(speckle_uint8 / 255.0, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    den_speckle = model(speckle_tensor.unsqueeze(0)).squeeze()

show_comparison(
    "Ruido Speckle",
    original_np,
    tensor_to_np(speckle_tensor),
    tensor_to_np(den_speckle)
)

compute_metrics(original_np, tensor_to_np(speckle_tensor), tensor_to_np(den_speckle))

print("LISTO!")
