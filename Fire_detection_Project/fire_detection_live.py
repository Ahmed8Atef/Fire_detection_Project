import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
import time

# Load model
model = load_model('best_fire_model.h5') 

# Load alarm sound
pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\HP\Downloads\OfccQO0BP7A.mp3")  # عدل المسار لو الملف اتحرك

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Error: Cannot access camera.")
    exit()

# Settings
img_size = (224, 224)
last_alert_time = 0
alert_interval = 5  # ثواني بين كل إنذار وإنذار

while True:
    ret, frame = cap.read()
    if not ret:
        print('❌ Error grabbing frame.')
        break

    # Preprocess image
    img = cv2.resize(frame, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    fire_probability = prediction[0][0]

    predicted_class = "Fire" if fire_probability > 0.9 else "No Fire"
    confidence = fire_probability * 100 if predicted_class == "Fire" else (1 - fire_probability) * 100

    # Display prediction
    label = f"{predicted_class} ({confidence:.2f}%)"
    color = (0, 0, 255) if predicted_class == "Fire" else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fire Detection", frame)

    # Play alarm if fire is detected
    current_time = time.time()
    if predicted_class == "Fire":
        if current_time - last_alert_time > alert_interval:
            pygame.mixer.music.play()
            last_alert_time = current_time
    else:
        pygame.mixer.music.stop()

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
