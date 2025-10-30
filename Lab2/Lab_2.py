import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# --- 1. Завантаження зображення ---
image_path = (
    "C:\\Users\\Mishanya\\PycharmProjects\\ComputerVision_labs\\Lab2\\imageKPI.png"
)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 2. Кольорова обробка ---
# Негатив для інверсії яскравості
negative = 255 - image_rgb

# Переведення в градації сірого
gray = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY)

# --- Підвищення контрастності ---
gray_eq = cv2.convertScaleAbs(gray, alpha=1.4, beta=-60)


# --- 3. Векторизація --
edges = cv2.Canny(gray_eq, 20, 300)

# --- 4. Контури ---
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image_rgb.copy()

buildings = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 400:  # прибираємо зовсім дрібні об’єкти, не використовуючи морфологію
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    # прямокутні форми (типові для будівель)
    if 0.4 < w / h < 3.0:
        buildings.append((x, y, w, h))
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            contour_img,
            str(len(buildings)),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

print(f"Знайдено будівель: {len(buildings)}")

# --- 5. Візуалізація ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Оригінал")
axs[0, 0].axis("off")

axs[0, 1].imshow(negative)
axs[0, 1].set_title("Негатив")
axs[0, 1].axis("off")

axs[1, 0].imshow(edges, cmap="gray")
axs[1, 0].set_title("Canny Edge Detection")
axs[1, 0].axis("off")

axs[1, 1].imshow(contour_img)
axs[1, 1].set_title(f"Векторизація будівель (кількість: {len(buildings)})")
axs[1, 1].axis("off")

plt.tight_layout()
plt.show()
