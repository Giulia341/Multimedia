import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_video_metrics(frames):
    """
    Calcola le metriche video (PSNR, MSE e SSIM) su una sequenza di frame.

    Args:
        frames: una lista di array NumPy che rappresentano i frame video.

    Returns:
        Una tupla contenente (media_mse, media_psnr, media_ssim), o None 
        se ci sono meno di 2 frame.
    """

    total_psnr = 0
    total_mse = 0
    total_ssim = 0

    frame_count = len(frames)

    # Verifica che ci siano almeno due frame per calcolare le medie
    if frame_count < 2:
        print("Non ci sono abbastanza frame per calcolare le metriche.")
        return None, None, None

    # Calcolo PSNR, MSE tra frame adiacenti
    for i in range(1, frame_count - 1):
        mse = np.mean((frames[i + 1] - frames[i]) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        total_mse += mse
        total_psnr += psnr

        # Calcolo del SSIM tra due frame adiacenti (conversione in scala di grigi)
        ssim_score, _ = ssim(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                             full=True)
        total_ssim += ssim_score

    # Calcolo della media dei valori per ogni metrica (escluso il primo frame)
    avg_mse = total_mse / (frame_count - 1)
    avg_psnr = total_psnr / (frame_count - 1)
    avg_ssim = total_ssim / (frame_count - 1)

    return avg_mse, avg_psnr, avg_ssim

