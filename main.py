import cv2
from tracker import ObjectTracker
from metrics import calculate_video_metrics
from post_processing import apply_blur, correct_color, adjust_brightness_contrast, apply_mask

# Creazione dell'oggetto tracker
tracker = ObjectTracker()

# Apertura del video
cap = cv2.VideoCapture('/Users/giulia/Multimedia_progetto/video/auto_2.mp4')

# Creazione dell'oggetto per il rilevamento degli oggetti
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40)

# Parametri per il rilevamento degli oggetti
#min_area = 300  #Dimensione minima dell'area
#shadow_threshold = 50  #Soglia per l'identificazione delle ombre

# Inizializzazione dei frame precedenti per PSNR, MSE e SSIM
prev_frame = None
prev_roi = None
frames = []

debug_printed = False  # Dichiarazione e inizializzazione della variabile debug_printed
while True:
    # Lettura del frame
    ret, frame = cap.read()
    if not ret:
        break

    # Estrazione della regione di interesse
    #roi = frame[340: 720, 500: 800] (quella iniziale)
    roi = frame[200:600, 300:700]

    
    # Memorizzazione del frame per il calcolo delle metriche
    frames.append(roi.copy())
    # Memorizzazione del frame corrente per il prossimo ciclo
    prev_frame = frame.copy()
    prev_roi = roi.copy()
    
    # 1. Rilevamento degli oggetti
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calcolo dell'area e rimozione degli elementi piccoli e delle ombre
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Tracking degli oggetti
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, obj_id = box_id
        cv2.putText(roi, str(obj_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Chiamata alle funzioni di post-elaborazione + debugging
    if not debug_printed: #se levo questo if allora mi permette di visualizzare questi filtri per l'intero video e mi stamperà il risultato per ogni frame
        blurred_frame = apply_blur(frame)
        print("Blurred Frame Shape:", blurred_frame.shape)  # Stampa frame sfocato
        corrected_frame = correct_color(frame)
        print("Corrected Frame Shape:", corrected_frame.shape)  # Stampa frame corretto nel colore
        adjusted_frame = adjust_brightness_contrast(frame, alpha=1.5, beta=10)
        print("Adjusted Frame Shape:", adjusted_frame.shape)  # Stampa frame con luminosità e contrasto regolati
        masked_frame = apply_mask(frame)
        print("Masked Frame Shape:", masked_frame.shape)  # Stampa frame mascherato
        debug_printed = True
    
    # Visualizzazione dei frame post-processing
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Blurred Frame", blurred_frame)
    cv2.imshow("Corrected Frame", corrected_frame)
    cv2.imshow("Adjusted Frame", adjusted_frame)
    cv2.imshow("Masked Frame", masked_frame)

    # Ingrandisce la finestra della maschera ROI
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 300)  # Impostare le dimensioni desiderate

    # Ridimensiona anche l'immagine della maschera per adattarla alla finestra
    mask_resized = cv2.resize(mask, (400, 300))  # Impostare le stesse dimensioni della finestra!

    cv2.imshow("Mask", mask_resized)
    
    # Interruzione se viene premuto ESC
    key = cv2.waitKey(30)
    if key == 27:
        break

# Rilascio delle risorse
cap.release()
cv2.destroyAllWindows()

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