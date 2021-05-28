import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import pafy
import cv2 as cv
import numpy as np

width, height = (1280, 736)

H = np.array([[-2.09332327e+00, -4.75764325e+00,  3.33436767e+02],
              [-2.24899272e+00, -1.85394832e+01,  2.86553095e+03],
              [-2.31176116e-03, -2.07072474e-02,  1.00000000e+00]], np.float64)


def transform_images(x_train, sizex, sizey):
    x_train = tf.image.resize(x_train, (sizex, sizey))
    x_train = x_train / 255
    return x_train


def convertPixel(x, y):
    dart_loc_temp = np.array([[x, y]], dtype="float32")
    dart_loc_temp = np.array([dart_loc_temp])
    dart_loc = cv.perspectiveTransform(dart_loc_temp, H)
    new_dart_loc = tuple(dart_loc.reshape(1, -1)[0])
    cv.circle(map, (round(new_dart_loc[0]), round(
        new_dart_loc[1])), 5, (0, 255, 0), -1)


url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv.VideoCapture(best.url)

map = cv.imread('./assets/map.PNG', -1)

model = YOLOv4(
    input_shape=(width, height, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
)

model.load_weights('./object_detection/yolov4.h5')

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

counter = 0

while True:
    # if frame is read correctly ret is True
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    counter = counter + 1
    if counter % 5 == 0:
        img_in = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, width, height)

        boxes, scores, classes, detections = model.predict(img_in)

        boxes = boxes[0] * [width, height, width, height]
        scores = scores[0]
        classes = classes[0].astype(int)
        detections = detections[0]

        for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
            convertPixel(round(xmin + xmax / 2), round(ymin + ymax / 2))
            if score > 0:
                if CLASSES[class_idx] in ['car', 'motorcycle', 'bus', 'truck']:
                    # create bounding box (frame of the video)
                    cv.rectangle(frame, (int(xmin), int(ymin)),
                                 (int(xmax), int(ymax)), (0, 0, 0), 2)
                    # create rectangle over the bounding box for the track id
                    cv.rectangle(frame, (int(xmin), int(
                        ymin) - 30), (int(xmin) + (len(CLASSES[class_idx])) * 1, int(ymin)), (255, 0, 0),
                        -1)
                    # put text in secound rectangle
                    cv.putText(frame, CLASSES[class_idx],
                               (int(xmin), int(ymin - 10)), 0, 0.75, (255, 255, 255), 2)

        cv.imshow('KIVO', frame)
        cv.imshow('Map', map)

    if cv.waitKey(25) == 27:
        break
    # When everything done, release the capture

cap.release()
cv.destroyAllWindows()
