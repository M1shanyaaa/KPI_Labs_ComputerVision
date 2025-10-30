import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Lab (Tkinter GUI)")
        self.root.geometry("420x700")
        self.root.configure(bg="#202020")

        # --- Завантаження зображення ---
        self.image_path = "imageKPI.png"
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            print("Помилка: не вдалося відкрити зображення.")
            exit()

        scale = 0.5
        new_size = (
            int(self.original.shape[1] * scale),
            int(self.original.shape[0] * scale),
        )
        self.img_resized = cv2.resize(self.original, new_size)
        self.hsv = cv2.cvtColor(self.img_resized, cv2.COLOR_BGR2HSV)

        # --- Змінні ---
        self.h_min = tk.IntVar(value=0)
        self.h_max = tk.IntVar(value=61)
        self.s_min = tk.IntVar(value=35)
        self.s_max = tk.IntVar(value=255)
        self.v_min = tk.IntVar(value=40)
        self.v_max = tk.IntVar(value=255)
        self.block_size = tk.IntVar(value=201)
        self.asp_min = tk.DoubleVar(value=0.1)
        self.asp_max = tk.DoubleVar(value=5.02)

        # --- Побудова інтерфейсу ---
        self.build_ui()

    def build_ui(self):
        def add_slider(label, var, from_, to_, step=1):
            frame = tk.Frame(self.root, bg="#202020")
            frame.pack(pady=5)
            lbl = tk.Label(
                frame, text=label, fg="white", bg="#202020", font=("Arial", 10, "bold")
            )
            lbl.pack()
            slider = tk.Scale(
                frame,
                from_=from_,
                to=to_,
                orient="horizontal",
                variable=var,
                length=300,
                resolution=step,
                bg="#303030",
                fg="white",
                troughcolor="#505050",
            )
            slider.pack()

        add_slider("Hue Min", self.h_min, 0, 179)
        add_slider("Hue Max", self.h_max, 0, 179)
        add_slider("Saturation Min", self.s_min, 0, 255)
        add_slider("Saturation Max", self.s_max, 0, 255)
        add_slider("Value Min", self.v_min, 0, 255)
        add_slider("Value Max", self.v_max, 0, 255)
        add_slider("Block Size", self.block_size, 3, 201, 2)
        add_slider("Aspect Ratio Min", self.asp_min, 0.1, 2.0, 0.05)
        add_slider("Aspect Ratio Max", self.asp_max, 1.0, 10.0, 0.05)

        tk.Button(
            self.root,
            text="Оновити обробку",
            command=self.update_processing,
            font=("Arial", 11, "bold"),
            bg="#0078D7",
            fg="white",
            activebackground="#3399ff",
            relief="ridge",
            width=20,
        ).pack(pady=20)

    def show_fullscreen(self, title, img):
        screen_res = 1280, 720
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        resized = cv2.resize(img, new_size)
        cv2.imshow(title, resized)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    def update_processing(self):
        img = self.img_resized.copy()

        # Початкове зображення
        self.show_fullscreen("1. Початкове зображення", img)

        # 2HSV маскування
        lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        mask = cv2.inRange(self.hsv, lower, upper)
        inv_mask = cv2.bitwise_not(mask)
        masked = cv2.bitwise_and(img, img, mask=inv_mask)
        self.show_fullscreen("2. Маскування HSV", masked)

        # 3️Бінаризація
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blk = self.block_size.get()
        if blk % 2 == 0:
            blk += 1
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 2
        )
        self.show_fullscreen("3. Бінаризація", binary)

        # 4️Морфологічна обробка
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        self.show_fullscreen("4. Морфологічна обробка", morph)

        # 5️онтури та фінальний результат
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area, max_area = 165, 12000
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / h
            if (
                min_area < area < max_area
                and self.asp_min.get() < aspect < self.asp_max.get()
            ):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1

        cv2.putText(
            img,
            f"Objects: {count}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        self.show_fullscreen("5. Фінальний результат", img)


# --- Запуск програми ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
