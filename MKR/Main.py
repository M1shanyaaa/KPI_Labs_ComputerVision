import cv2
import numpy as np

# 1 Завантаження стереозображень
left = cv2.imread("left.png")
right = cv2.imread("right.png")

if left is None or right is None:
    print("Помилка: зображення не знайдені! Перевірте наявність left.png та right.png")
    exit()

# Масштабування правого зображення під розміри лівого
h, w = left.shape[:2]
right = cv2.resize(right, (w, h))

# 2. Виявлення ключових точок ORB та пошук відповідностей
orb = cv2.ORB_create(1500)
kp1, des1 = orb.detectAndCompute(left, None)
kp2, des2 = orb.detectAndCompute(right, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:60]

# 3. Обчислення паралаксу (горизонтального зміщення)
parallaxes = []
for m in good_matches:
    (x1, _) = kp1[m.queryIdx].pt
    (x2, _) = kp2[m.trainIdx].pt
    parallaxes.append(x2 - x1)

parallaxes = np.array(parallaxes)
mean_parallax = np.mean(parallaxes)
parallax_abs = abs(mean_parallax)

# 4. Виведення аналізу
print(f"Виявлено відповідностей ключових точок: {len(matches)}")
print(f"Середній паралакс (px): {mean_parallax:.2f}")

if parallax_abs < 3:
    print("3D-ефект буде дуже слабким. Збільшіть відстань між камерами.")
elif 3 <= parallax_abs <= 30:
    print("Рекомендований паралакс: очікується хороший 3D-ефект!")
elif 30 < parallax_abs <= 80:
    print("Сильний паралакс: можливий дискомфорт для очей, бажано зменшити відстань.")
else:
    print("Надто великий паралакс: 3D-ефект виглядатиме некомфортно та двоїтиметься.")

# Створення анагліфа 3D
# R-канал з лівого зображення
r = left[:, :, 2]
# G та B канали з правого
g = right[:, :, 1]
b = right[:, :, 0]

anaglyph = cv2.merge((b, g, r))
cv2.imwrite("anaglyph.png", anaglyph)

print("Анагліф створено та збережено як anaglyph.png")

# Візуалізація результатів
match_vis = cv2.drawMatches(left, kp1, right, kp2, good_matches, None, flags=2)

cv2.imshow("Відповідності ключових точок", match_vis)
cv2.imshow("Анагліф 3D", anaglyph)

cv2.waitKey(0)
cv2.destroyAllWindows()
