import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import os, time
import requests

#  SETTINGS
use_video_file = True

video_filename = "video_security_camera.mp4"
# video_filename = "video_security_camera2.mp4"

CONFIDENCE_THRESHOLD = 0.25
YOLO_EVERY_N_FRAMES = 5

TELEGRAM_ENABLED = True
BOT_TOKEN = "8542134191:AAGzCeFOxmN8zrlczNaEfi38jx-zaIbYOE0"
CHAT_ID = "1009134562"
LONG_MOVEMENT_SECONDS = 10


#  TELEGRAM
def send_telegram_photo(image, caption="🚨 Виявлено рух людини!"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    _, img_encoded = cv2.imencode(".jpg", image)
    files = {"photo": ("alert.jpg", img_encoded.tobytes())}
    data = {"chat_id": CHAT_ID, "caption": caption}
    requests.post(url, data=data, files=files)


#  YOLO
model = YOLO("yolov8n.pt")

#  VIDEO
cap = cv2.VideoCapture(video_filename if use_video_file else 0)

#  Motion (KNN)
fg_model = cv2.createBackgroundSubtractorKNN(
    history=500, dist2Threshold=400, detectShadows=True
)

frame_count = 0
tracked_people = []
person_timeout = 0
motion_state = False

movement_start_time = None
last_telegram_photo_time = 0

print("Система активована")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Night Enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(3.0).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #  Motion Detection
    fg_mask = fg_model.apply(blur)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 7)
    motion_pixels = np.sum(thresh == 255)

    motion_detected = motion_pixels > 2500

    #  YOLO
    if frame_count % YOLO_EVERY_N_FRAMES == 0:
        small = cv2.resize(frame, (640, 360))
        results = model(small, verbose=False, agnostic_nms=True)

        new_people = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                if cls == 0 and conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 * frame.shape[1] / 640)
                    y1 = int(y1 * frame.shape[0] / 360)
                    x2 = int(x2 * frame.shape[1] / 640)
                    y2 = int(y2 * frame.shape[0] / 360)
                    new_people.append((x1, y1, x2, y2))

        if new_people:
            tracked_people = new_people
            person_timeout = 15
        else:
            person_timeout = max(0, person_timeout - 1)

    #  Draw people
    if person_timeout > 0:
        for x1, y1, x2, y2 in tracked_people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #  LOGIC
    if motion_detected and person_timeout > 0:
        if not motion_state:
            print("🚨 Рух людини!")
            winsound.Beep(1500, 200)
            motion_state = True
            movement_start_time = time.time()
            last_telegram_photo_time = movement_start_time

            if TELEGRAM_ENABLED:
                send_telegram_photo(frame, "🚨 Рух розпочався!")

        else:
            # long motion photo
            now = time.time()
            if now - last_telegram_photo_time > LONG_MOVEMENT_SECONDS:
                print("📸 Надсилаю нове фото — рух триває довго!")
                if TELEGRAM_ENABLED:
                    send_telegram_photo(
                        frame, f"📸 Рух триває більше {LONG_MOVEMENT_SECONDS} секунд!"
                    )
                last_telegram_photo_time = now

    else:
        # RESET WHEN MOTION ENDS
        if motion_state:
            print("Рух закінчився")
        motion_state = False
        movement_start_time = None

    # Show windows
    cv2.imshow("Monitoring System", frame)
    cv2.imshow("Motion Map", thresh)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
