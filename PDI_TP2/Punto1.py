import cv2
import matplotlib.pyplot as plt
# Load the image


img = cv2.imread('C:/Users/tomyl/OneDrive/Escritorio/Procesamiento de Imagenes Digitales/venv/Img/rosa.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('C:/Users/tomyl/OneDrive/Escritorio/Procesamiento de Imagenes Digitales/venv/Img/building.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('C:/Users/tomyl/OneDrive/Escritorio/Procesamiento de Imagenes Digitales/venv/Img/craneo.jpg', cv2.IMREAD_COLOR)

def subsampling_imagen():
    
    print(f"📏 Tamaño original: {img.shape}")

    if img.shape != (1024, 1024):
        img_original = cv2.resize(img, (1024, 1024))
        print("📏 Redimensionado a 1024x1024")
    else:
        img_original = img.copy()
    
    # 4. APLICAR SUBSAMPLING PASO A PASO
    print("\n🔧 Aplicando subsampling...")
    
    # Factor de 2 en cada paso (eliminar 1 de cada 2 píxeles)
    img_512 = img_original[::2, ::2]      # 1024 → 512
    img_256 = img_512[::2, ::2]           # 512 → 256  
    img_128 = img_256[::2, ::2]           # 256 → 128
    img_64 = img_128[::2, ::2]            # 128 → 64
    img_32 = img_64[::2, ::2]             # 64 → 32
    
    # Lista de imágenes y sus tamaños
    imagenes = [img_original, img_512, img_256, img_128, img_64, img_32]
    tamaños = [1024, 512, 256, 128, 64, 32]
    factores = [1, 2, 4, 8, 16, 32]
    
    # 5. MOSTRAR INFORMACIÓN
    print("\n📊 Resultados del subsampling:")
    for i, (tamaño, factor) in enumerate(zip(tamaños, factores)):
        print(f"  Paso {i}: {tamaño}x{tamaño} (Factor ÷{factor})")
    
    # 6. VISUALIZAR RESULTADOS
    
    return imagenes, tamaños

def subsample_to_48x64(img):
    
    # Reduce la imagen de 192x256 a 48x64 usando subsampling.
    
    img = cv2.resize(img, (256, 192))  # Redimensionar a 192x256
    
    # Verifica tamaño esperado
    if img.shape[0] != 192 or img.shape[1] != 256:
        raise ValueError("La imagen debe ser de tamaño 192x256")
    # Toma 1 de cada 4 píxeles en ambas dimensiones
    img_subsampled = img[::4, ::4]
    return img_subsampled

# Función para reducir niveles de gris
def reduce_gray_levels (): 
    img_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img_gray128 = (img_gray // 2) * 2
    img_gray64 = (img_gray // 4) * 4
    img_gray32 = (img_gray // 8) * 8
    img_gray16 = (img_gray // 16) * 16
    img_gray8 = (img_gray // 32) * 32
    img_gray4 = (img_gray // 64) * 64
    img_gray2 = (img_gray // 128) * 128

    imgs = [img_gray, img_gray128, img_gray64, img_gray32, img_gray16, img_gray8, img_gray4, img_gray2]

    return imgs


# Subsampling de la imagen a 48x64
img2_subsampled = subsample_to_48x64(img2)

# Subsampling paso a paso
subsampling_imagen()

imgenes = reduce_gray_levels()

# Visualización de todas las imágenes del subsampling
plt.figure(figsize=(12, 8))
for i, img in enumerate(subsampling_imagen()[0]):
    plt.subplot(2, 3, i + 1)
    plt.title(f"Paso {i} - {subsampling_imagen()[1][i]}x{subsampling_imagen()[1][i]}")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.suptitle("Subsampling de Imagen", fontsize=16)

# Visualización de la imagen reducida a 48x64 junto a la original de 192x256
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen a 48x64")
plt.axis('off')
plt.imshow(cv2.cvtColor(img2_subsampled, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Imagen  a 192x256")
plt.axis('off')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

# Visualización de todas las imágenes del reduce gray levels
plt.figure(figsize=(16, 8))
for i, img in enumerate(imgenes):
    plt.subplot(2, 4, i + 1)
    niveles = [256, 128, 64, 32, 16, 8, 4, 2]
    plt.title(f"Niveles de gris: {niveles[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')

plt.show()