import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PRAKTIKUM PENGOLAHAN CITRA DIGITAL - SESI 2
# ============================================================

INPUT_IMAGE = "input/images.jpg"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. MEMBACA GAMBAR
# ============================================================

img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {INPUT_IMAGE}\n"
        "Pastikan file gambar berada di folder input."
    )

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("========================================")
print("INFORMASI CITRA")
print("========================================")
print("Tipe data :", type(img))
print("Shape     :", img.shape)

# ============================================================
# 2. MENAMPILKAN GAMBAR
# ============================================================

plt.figure(figsize=(6, 5))
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")
plt.show()

# ============================================================
# 3. MENENTUKAN UKURAN GAMBAR
# ============================================================

height, width = img.shape[:2]

print("Tinggi gambar :", height, "piksel")
print("Lebar gambar  :", width, "piksel")
print("Total piksel  :", height * width)

# ============================================================
# 4. MENENTUKAN JUMLAH CHANNEL
# ============================================================

if len(img.shape) == 2:
    channels = 1
else:
    channels = img.shape[2]

print("Jumlah channel:", channels)

# ============================================================
# 5. MEMISAHKAN CHANNEL RGB
# ============================================================

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(R, cmap="Reds")
plt.title("Red Channel")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(G, cmap="Greens")
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(B, cmap="Blues")
plt.title("Blue Channel")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 6. MENGUBAH RGB -> GRAYSCALE
# ============================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("\n========================================")
print("GRAYSCALE")
print("========================================")
print("Shape grayscale      :", gray.shape)
print("Intensitas minimum   :", gray.min())
print("Intensitas maksimum  :", gray.max())

plt.figure(figsize=(6, 5))
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Grayscale")
plt.axis("off")
plt.show()

# ============================================================
# 7. KUANTISASI 8, 4, DAN 2 BIT
# ============================================================

def quantize(image, bits):
    """
    Mengurangi jumlah tingkat intensitas citra.
    8 bit = 256 level
    4 bit = 16 level
    2 bit = 4 level
    """
    levels = 2 ** bits
    step = 256 / levels

    result = np.floor(image / step) * step
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


gray_8bit = quantize(gray, 8)
gray_4bit = quantize(gray, 4)
gray_2bit = quantize(gray, 2)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_8bit, cmap="gray", vmin=0, vmax=255)
plt.title("Kuantisasi 8 Bit (256 Level)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray_4bit, cmap="gray", vmin=0, vmax=255)
plt.title("Kuantisasi 4 Bit (16 Level)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gray_2bit, cmap="gray", vmin=0, vmax=255)
plt.title("Kuantisasi 2 Bit (4 Level)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 8. GRAYSCALE -> BINER
# ============================================================

threshold_default = 127

_, binary = cv2.threshold(
    gray,
    threshold_default,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(6, 5))
plt.imshow(binary, cmap="gray", vmin=0, vmax=255)
plt.title(f"Citra Biner - Threshold {threshold_default}")
plt.axis("off")
plt.show()

# ============================================================
# 9. MENGUJI MINIMAL 4 NILAI THRESHOLD
# ============================================================

thresholds = [50, 100, 150, 200]

plt.figure(figsize=(16, 4))

for i, t in enumerate(thresholds, start=1):

    _, binary_t = cv2.threshold(
        gray,
        t,
        255,
        cv2.THRESH_BINARY
    )

    plt.subplot(1, 4, i)
    plt.imshow(binary_t, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Threshold = {t}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 10. MENYIMPAN HASIL PENGOLAHAN
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "01_grayscale.jpg"),
    gray
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_kuantisasi_4bit.jpg"),
    gray_4bit
)

cv2.imwrite(
    str(OUTPUT_DIR / "03_kuantisasi_2bit.jpg"),
    gray_2bit
)

cv2.imwrite(
    str(OUTPUT_DIR / "04_binary_threshold_127.jpg"),
    binary
)

for t in thresholds:

    _, binary_t = cv2.threshold(
        gray,
        t,
        255,
        cv2.THRESH_BINARY
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"05_binary_threshold_{t}.jpg"),
        binary_t
    )

print("\n========================================")
print("PRAKTIKUM SELESAI")
print("========================================")
print("Hasil pengolahan tersimpan di:")
print(OUTPUT_DIR.resolve())
