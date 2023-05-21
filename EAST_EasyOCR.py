from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import shutil
import os
import easyocr
import googletrans

reader = easyocr.Reader(['en'])

translator = googletrans.Translator()

word = "Her"
word = word.lower()


res_text = translator.translate(word, src='en', dest='ko')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
                help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
count = 1

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("C:/Users/82106/Downloads/frozen_east_text_detection/frozen_east_text_detection.pb")

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    orig = frame.copy()

    #이미지 컬러로 읽기(EAST모델 회색 못 읽음)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #이미지 선명하게
    kernel = np.ones((3,3), dtype=np.float64)/9
    frame = cv2.filter2D(frame, -1, kernel)

    #가우시안 블러
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    #이미지 대비 높이기
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    #이미지 컬로로 변환
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)



    if not ret:
        break

    (H, W) = frame.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # change box width and height -> positive will add pixels and vice-versa
    box_width_padding = 3
    box_height_padding = 3

    croppedList = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:

        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW) - box_width_padding
        startY = int(startY * rH) - box_height_padding
        endX = int(endX * rW) + box_width_padding
        endY = int(endY * rH) + box_height_padding

        # # draw the bounding box on the image
        # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        #영역 추출
        text_region = orig[startY:endY, startX:endX]
        
        try:
            # 색상 GRAY로 변경
            text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

            # 이미지의 노이즈 제거
            text_region = cv2.GaussianBlur(text_region, (5, 5), 0)

            text = reader.readtext(text_region, detail = 0)

            text = text[0].lower()
            
            # cv2.imshow('roi', text_region)
            
            if word in text:

                # draw the bounding box on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # #글자 삽입
                # cv2.putText(orig, text[0], (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)

                print(res_text.text)

        except:
            continue
    # Display the output
    cv2.imshow("Text Detection", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
