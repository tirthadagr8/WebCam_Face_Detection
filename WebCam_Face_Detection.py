import cv2

Detection_Algo = cv2.CascadeClassifier('Frontal_Face.xml')

webcam = cv2.VideoCapture(0)

while True:
    success, img = webcam.read()
    Grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Coordinates = Detection_Algo.detectMultiScale(Grayscale_img)
    for (x, y, w, h) in Coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("WebCam", img)
    cv2.waitKey(1)
