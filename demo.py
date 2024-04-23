import argparse
import os
import time
import numpy as np
import cv2
from model.simpleModel import ConvNet2, ConvNet3, ConvNet4
from model.vgg import Vgg16
import tensorflow as tf

# NOTE: IMPORTANT DISCLAIMER. THIS IS A DEMO FILE THE WE LARGELY SOURCED FROM THE FOLLOWING REPO: https://github.com/cansik/yolo-hand-detection/blob/master/yolo.py
# We have made some modifications to the code to suit our needs. We have also added some additional functionality to the code.
# We also relied on https://github.com/cansik/yolo-hand-detection/blob/master/yolo.py to help us understand how to use the YOLO model.
# CREDIT TO Florian Bruggisser, djthegr8 DJ, and roachsinai RoachZhao for the original code.

class YOLO:

    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.output_names = []
        self.labels = labels
        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
        except:
            raise ValueError("Couldn't find the models!\nDid you forget to download them manually (and keep in the "
                             "correct directory, models/) or run the shell script?")

        ln = self.net.getLayerNames()
        for i in self.net.getUnconnectedOutLayers():
            self.output_names.append(ln[int(i) - 1])

    def inference_from_file(self, file):
        mat = cv2.imread(file)
        return self.inference(mat)

    def inference(self, image):
        ih, iw = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.output_names)
        end = time.time()
        inference_time = end - start

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([iw, ih, iw, ih])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

        return iw, ih, inference_time, results

def bounding_box(image, detection):
    '''
    Extract the bounding box from the image to pass to the classifier
    '''
    id, name, confidence, x, y, w, h = detection
    # Scale to frame size with a margin for further classification
    w = int(w * 2)
    h = int(h * 2)

    w = max(w, h)
    h = max(w, h)

    x -= (w - detection[5]) // 2
    y -= (h - detection[6]) // 2

    return id, name, confidence, x, y, w, h, image[y:y+h, x:x+w]

def transform_image(image):
    # TODO: Implement image transformations

    tf_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tf_image = tf.image.resize(tf_image, (32, 32))
    tf_image = tf_image / 255.0

    return tf_image

def main():
    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
    
    # path to weights
    convnet = os.path.join("temp", "convnet3_adam_categorical_crossentropy_lr0.001_mo9e-06_rs4264", "convnet3.h5")
    vgg = os.path.join("temp", "vgg_adam_categorical_crossentropy_lr0.001_mo9e-06_rs9932", "vgg.h5")

    yolo = YOLO("temp/YOLO/cross-hands-yolov4-tiny.cfg", "temp/YOLO/cross-hands-yolov4-tiny.weights", ["hand"])
    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    # Load the model for classification
    model = Vgg16(32, 'rgb', 29)#ConvNet3(64, num_classes=29, input_shape=(64, 64, 3))
    # Debugged building: https://stackoverflow.com/questions/59356410/tensorflow-2-0-build-function
    # for an explanation of why you need to call .build() before loading weights.
    model.build((None, 32, 32, 3))
    # Load the weights into the model
    model.load_weights(vgg)

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        exit(1)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break

        width, height, inference_time, results = yolo.inference(frame)

        # Sort by confidence
        results.sort(key=lambda x: x[2], reverse=True)

        # Number of hands detected
        hand_count = len(results)

        # Display the results
        for detection in results:
            # Scale to frame size with a margin for further classification
            id, name, confidence, x, y, w, h, bbox = bounding_box(frame, detection)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Classify the hand
            try:
                bbox = transform_image(bbox)
                bbox = tf.expand_dims(bbox, axis=0)
                softmax_preds = model.predict(bbox)
                prediction = np.argmax(softmax_preds)
                label = classes[prediction]
                cv2.putText(frame, f"{label}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                pass
            
        cv2.putText(frame, f"Time: {inference_time:.2f} | Hands: {hand_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('-confidence', type=float, default=0.5, help='Confidence')
    parser.add_argument('-threshold', type=float, default=0.3, help='Threshold')
    parser.add_argument('-size', type=int, default=500, help='Size')
    args = parser.parse_args()
    main()