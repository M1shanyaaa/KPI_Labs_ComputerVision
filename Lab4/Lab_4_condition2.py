import cv2
import numpy as np

# Ініціалізація відео
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Камеру не вдалося відкрити.")
    exit()

cv2.namedWindow("CamShift Tracker")

roi_hist = None
track_window = None
selection = None
drag_start = None
tracking = False
frame_in_mouse = None


def onmouse(event, x, y, flags, param):
    global selection, drag_start, tracking, track_window, roi_hist, frame_in_mouse

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        selection = None
        tracking = False

    elif event == cv2.EVENT_MOUSEMOVE and drag_start:
        xi, yi = drag_start
        x0, y0 = min(xi, x), min(yi, y)
        x1, y1 = max(xi, x), max(yi, y)
        selection = (x0, y0, x1, y1)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None
        if selection is not None:
            x0, y0, x1, y1 = selection
            w, h = x1 - x0, y1 - y0
            if w > 0 and h > 0 and frame_in_mouse is not None:
                roi = frame_in_mouse[y0:y1, x0:x1]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(
                    hsv_roi,
                    np.array((0.0, 60.0, 32.0)),
                    np.array((180.0, 255.0, 255.0)),
                )
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                track_window = (x0, y0, w, h)
                tracking = True


cv2.setMouseCallback("CamShift Tracker", onmouse)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося зчитати кадр з камери")
        break

    frame = cv2.flip(frame, 1)
    frame_in_mouse = frame.copy()
    vis = frame.copy()

    if selection and not tracking:
        x0, y0, x1, y1 = selection
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

    if tracking and roi_hist is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
        )
        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        backproj &= mask

        ret, track_window = cv2.CamShift(backproj, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    cv2.putText(
        vis,
        "Select ROI with mouse | ESC - exit | R - reset",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.imshow("CamShift Tracker", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("r"):
        selection = None
        tracking = False
        roi_hist = None
        track_window = None

cap.release()
cv2.destroyAllWindows()
