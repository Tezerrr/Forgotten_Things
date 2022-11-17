import cv2
import numpy as np

# Подготовка к получению контуров
def preProcessing(img):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200,200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres


# img = cv2.imread("Data/shapeimg.png")
# img = cv2.imread("Data/imShp.jpg")

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    global COORDS
    best = np.array(list())
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000<=area<=16000:
            # cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            COORDS.append([x, y, w, h])
            if area > maxArea and 4 <= objCor <= 8:
                best = approx
                maxArea = area
            # Проверяем движение объекта, сравнивая измененния его координат
            if (abs(COORDS[1][0] - COORDS[0][0]) <= 3) and (abs(COORDS[1][1] - COORDS[0][1]) <= 3) and (
                    abs(COORDS[1][2] - COORDS[0][2]) <= 3) and (abs(COORDS[1][3] - COORDS[0][
                3]) <= 3):
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame1, 'Забытая вещь',
                            (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (255, 255, 255), 2)
            # Удаляем старые коориднаты
            COORDS.pop(0)
            return best


# imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgContour = img.copy()
# imgBlur = cv2.GaussianBlur(imGray, (7,7),1)
# imgCanny = cv2.Canny(imgBlur, 50,50)
# getContours(imgCanny)
# imgStack = stackImages(0.4,([img, imGray,imgContour], [imgCanny, imgBlur,imgBlur]))
# #imgStack = imgContour
# cv2.imshow("Stack",imgStack)
# cv2.waitKey(0)

movie = cv2.VideoCapture("C://Users/Tezer/Downloads/Telegram Desktop/YT4.mp4")
width = int(movie.get(3))
height = int(movie.get(4))
print(movie.get(1), movie.get(2))
size = (width, height)
count = 0
ret, frame = movie.read()
movie.set(1, 10)
COORDS = [[0, 0, 0, 0]]
writer = cv2.VideoWriter("Result4.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 10, size)
while movie.isOpened():
    ret, frame1 = movie.read()
    if ret:
        sub = cv2.subtract(frame, frame1)
        subThres = preProcessing(sub)
        getContours(subThres)
        writer.write(frame1)
        cv2.imshow("Result44", frame1)
        if cv2.waitKey(10) == ord("q"):
            break
        count += 3
        movie.set(1, count)
    else:
        movie.release()
        break
