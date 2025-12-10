import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torch.serialization

#########################################
# 1. Modelo DnCNN
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
        return x - out  # residual learning


#########################################
# 2. Cargar modelo entrenado
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

torch.serialization.add_safe_globals([DnCNN])
model = torch.load("model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

transform = T.ToTensor()


#########################################
# 3. DETECCIÓN DEL TIPO DE RUIDO
#########################################
def detectar_tipo_ruido(img_gray_01: np.ndarray) -> str:
    
    flat = img_gray_01.ravel()

    # 1) Sal y Pimienta → muchos píxeles muy cercanos a 0 o 1
    extremos = np.mean((flat < 0.02) | (flat > 0.98))
    # umbral ajustable
    if extremos > 0.01:
        return "sp"

    # 2) Gaussiano → residual ~ distribución normal
    blur = cv2.GaussianBlur(img_gray_01, (5, 5), 1.0)
    residual = img_gray_01 - blur
    r = residual.ravel()
    media = np.mean(r)
    std = np.std(r) + 1e-8
    z = (r - media) / std
    kurtosis = np.mean(z ** 4)  # normal ideal ≈ 3

    if 2.0 < kurtosis < 5.0:
        return "gauss"

    # 3) Si no encaja bien → lo tomamos como speckle / otro
    return "speckle"


#########################################
# 4. MÉTODOS DE DENOISING SEGÚN TIPO
#########################################
def denoise_gauss_dncnn(img_gray_01: np.ndarray) -> np.ndarray:
    """Usa DnCNN para ruido gaussiano"""
    img_tensor = transform((img_gray_01 * 255).astype(np.uint8))  # [1,H,W], 0-1
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        den = model(img_tensor).cpu().squeeze(0).squeeze(0).numpy()

    den = np.clip(den, 0, 1)
    return den


def denoise_sal_pimienta_mediana(img_gray_01: np.ndarray) -> np.ndarray:
    """Mediana (funciona muy bien para sal y pimienta)"""
    img_uint8 = (img_gray_01 * 255).astype(np.uint8)
    den = cv2.medianBlur(img_uint8, 3)  # podés probar 5
    return den.astype(np.float32) / 255.0


def denoise_speckle_log_gauss(img_gray_01: np.ndarray) -> np.ndarray:
    """
    Ruido speckle ~ multiplicativo.
    Truco clásico: log, suavizar, exp.
    """
    eps = 1e-6
    log_img = np.log(img_gray_01 + eps)
    log_smooth = cv2.GaussianBlur(log_img, (5, 5), 1.0)
    den = np.exp(log_smooth) - eps
    den = np.clip(den, 0, 1)
    return den


#########################################
# 5. MAIN: cargar imagen, detectar y filtrar
#########################################
def main():
    if len(sys.argv) < 2:
        print("Uso: python auto_denoise.py ruta_imagen")
        sys.exit(1)

    ruta = sys.argv[1]
    # leemos la imagen y la pasamos a gris
    img = Image.open(ruta).convert("L")
    img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]

    tipo = detectar_tipo_ruido(img_np)
    print("Tipo de ruido detectado:", tipo)

    if tipo == "gauss":
        den = denoise_gauss_dncnn(img_np)
        metodo = "DnCNN (ruido gaussiano)"
    elif tipo == "sp":
        den = denoise_sal_pimienta_mediana(img_np)
        metodo = "Mediana (ruido sal y pimienta)"
    else:
        den = denoise_speckle_log_gauss(img_np)
        metodo = "Gaussiano en dominio log (speckle)"

    # Mostrar resultado
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Imagen de entrada (ruidosa)")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Denoised\n{metodo}")
    plt.imshow(den, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Guardar resultado
    out_uint8 = (den * 255).astype(np.uint8)
    out_img = Image.fromarray(out_uint8)
    out_img.save("denoised_output.png")
    print("Imagen denoised guardada como denoised_output.png")


if __name__ == "__main__":
    main()
