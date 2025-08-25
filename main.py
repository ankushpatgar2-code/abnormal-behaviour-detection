import cv2
import os
import datetime
import pygame

print("🔧 Starting abnormal behavior detection...")

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Create alerts folder
if not os.path.exists("alerts"):
    os.makedirs("alerts")
    print("📁 Created alerts folder.")
else:
    print("📁 Alerts folder already exists.")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("❌ Cannot read initial frame.")
    cap.release()
    exit()

# Preprocess first frame
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    abnormal = False
    for c in contours:
        if cv2.contourArea(c) > 2000:
            abnormal = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if abnormal:
        pygame.mixer.music.play()
        filename = f"alerts/alert_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved alert: {filename}")

    cv2.imshow("Motion Detection", frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

