import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture("./Videos/leija.mp4")
    cap.set(10, 160)
    imgHeight = 640
    imgWidth = 480

    imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Termino Stream")
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (13, 13), 0)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2)
        #_, imgThreshold = cv2.threshold(imgBlur, 120, 255, cv2.THRESH_BINARY)
        imgCanny = cv2.Canny(imgThreshold, 0, 70)
        imgDilate = cv2.dilate(imgCanny, (13, 13), iterations=10)
        imgErode = cv2.erode(imgDilate, (13, 13), iterations=5)

        contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxcnt = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(maxcnt)
        #im2 = cv2.drawContours(imgBigContours, [maxcnt], -1, (0, 255, 0), thickness=-1)

        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))

        cv2.imshow('Result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
