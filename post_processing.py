import cv2
import numpy as np

# applica filtro di blurring
def apply_blur(frame):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred_frame

# applica filtro di correzione del colore
def correct_color(frame):
    corrected_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return corrected_frame

# applica filtro di miglioramento del contrasto
def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted_frame

# maschera di applicazione all'intero frame - implementa anche una forma di sfocatura
def apply_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (500, 340), (800, 720), 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame
