import c1
import c2
import os
import numpy as np
import cv2


os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")


engine=c1.engine

name_face=c2.name_face


item = []
final = []

img = cv2.imread("test.jpg")
height, width, channels = img.shape


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

layer_name = net.getLayerNames()

output_layers = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#detecting objects
blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)


net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes , confidences , 0.5 , 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)) :

                if i in indexes :
                    x , y , w , h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                    item.append(label)
final = list(set(item))                                                             #to remove the repetative words.
print(final)
a = "person"
if a in final:
    c2.face_recognition(img)
    c1.speak(name_face)
c1.speak(final)
cv2.waitKey(0)
cv2.destroyAllWindows()