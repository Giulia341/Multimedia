import cv2
import numpy as np
from metrics import calculate_video_metrics

cap = cv2.VideoCapture('/Users/giulia/Downloads/auto_6.mp4')

prev_frame = None
prev_roi = None
frames = []

# Lettura e memorizzazione dei frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Calcolo del PSNR e MSE medi
avg_mse, avg_psnr, avg_ssim = calculate_video_metrics(frames)

# Verifica se le metriche sono state calcolate con successo
if avg_mse is not None and avg_psnr is not None:
    # Visualizzazione dei risultati
    print("Media MSE:", avg_mse)
    print("Media PSNR:", avg_psnr)
    print("Media SSIM:", avg_ssim)
else:
    print("Impossibile calcolare le metriche. Assicurati di avere almeno due frame nel video.")
    
# Rilascio delle risorse
cap.release()
cv2.destroyAllWindows()