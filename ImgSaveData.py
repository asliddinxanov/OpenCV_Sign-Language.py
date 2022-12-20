import time
import cv2
import cvzone.HandTrackingModule
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = cvzone.HandTrackingModule.HandDetector(maxHands=1)

value_set = 20
imgSize = 300

# A〜F画像を保存するフォルダを作成
dataL_A = "Data/A(Left)"
dataR_A = "Data/A(Right)"
dataL_B = "Data/B(Left)"
dataR_B = "Data/B(Right)"
dataL_C = "Data/C(Left)"
dataR_C = "Data/C(Right)"
dataL_D = "Data/D(Left)"
dataR_D = "Data/D(Right)"
dataL_E = "Data/E(Left)"
dataR_E = "Data/E(Right)"
dataL_F = "Data/F(Left)"
dataR_F = "Data/F(Right)"

counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - value_set: y + h + value_set, x - value_set: x + w + value_set]

        imgCropShape = imgCrop.shape

        aspect = h / w

        if aspect > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        counter += 1
        cv2.imwrite(f'{dataL_A}/Image-{time.time()}.jpg', imgWhite)
    elif key == ord("b"):
        counter += 1
        cv2.imwrite(f'{dataL_B}/Image-{time.time()}.jpg', imgWhite)
    elif key == ord("c"):
        counter += 1
        cv2.imwrite(f'{dataL_C}/Image-{time.time()}.jpg', imgWhite)
    elif key == ord("d"):
        counter += 1
        cv2.imwrite(f'{dataL_D}/Image-{time.time()}.jpg', imgWhite)
    elif key == ord("e"):
        counter += 1
        cv2.imwrite(f'{dataL_E}/Image-{time.time()}.jpg', imgWhite)
    elif key == ord("f"):
        counter += 1
        cv2.imwrite(f'{dataL_F}/Image-{time.time()}.jpg', imgWhite)