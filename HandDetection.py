import cv2
import cvzone.HandTrackingModule

cap = cv2.VideoCapture(0)
detector = cvzone.HandTrackingModule.HandDetector(maxHands=1)

value_set = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgCrop = img[y:y + h, x:x + w]
        cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):
        break
