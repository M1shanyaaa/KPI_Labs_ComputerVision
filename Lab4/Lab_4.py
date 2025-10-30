import cv2 as cv
import numpy as np


def preprocess_image(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img, (800, 800))  # для порівнянності
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # --- Корекція контрасту CLAHE ---
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Фільтрація шумів ---
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    return img, gray


def match_images(img1_path, img2_path):
    # Попередня обробка
    img1, gray1 = preprocess_image(img1_path)
    img2, gray2 = preprocess_image(img2_path)

    # --- Harris кути + SIFT дескриптори ---
    harris1 = cv.cornerHarris(np.float32(gray1), 2, 3, 0.04)
    harris2 = cv.cornerHarris(np.float32(gray2), 2, 3, 0.04)

    kp1 = [
        cv.KeyPoint(float(x[1]), float(x[0]), 13)
        for x in np.argwhere(harris1 > 0.02 * harris1.max())
    ]
    kp2 = [
        cv.KeyPoint(float(x[1]), float(x[0]), 13)
        for x in np.argwhere(harris2 > 0.02 * harris2.max())
    ]

    sift = cv.SIFT_create()
    kp1, des1 = sift.compute(gray1, kp1)
    kp2, des2 = sift.compute(gray2, kp2)

    print(f"Ключові точки: {len(kp1)} (Good), {len(kp2)} (Bad)")

    # Порівняння ознак (FLANN matcher)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Тест співпадінь за Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.915 * n.distance and m.distance < 250:
            good_matches.append(m)

    # Розрахунок ймовірності ідентифікації
    P = len(good_matches) / ((len(kp1) + len(kp2)) / 2)
    print(f"Ймовірність ідентифікації: {P:.3f} (збігів: {len(good_matches)})")

    if len(good_matches) > 1000 or P > 0.1:
        print("Ідентифікація впевнена.")
    elif len(good_matches) > 500 or P > 0.05:
        print("Ідентифікація можлива, але не повна.")
    else:
        print("Ідентифікація слабка.")

    # Візуалізація
    result = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv.DrawMatchesFlags_DEFAULT,
    )

    cv.imshow("Comparison(SIFT+Harris)", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return P


if __name__ == "__main__":
    good = "Good_photo.png"
    bad = "Bad_photo.png"

    P = match_images(good, bad)
