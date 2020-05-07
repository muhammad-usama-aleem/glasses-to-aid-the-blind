import os
import numpy as np
import cv2
import pyttsx3
import face_recognition as fg
from PIL import Image, ImageDraw

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")

#YOLO

engine = pyttsx3.init()


def speak(text):
    engine.setProperty("rate", 100)

    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)

    engine.say(str(text))
    engine.runAndWait()


name_face = []


def face_recognition(img):
    usama_img = fg.load_image_file("usama.jpg")
    junaid_img = fg.load_image_file("junaid.jpg")

    usama_encoding = fg.face_encodings(usama_img)[0]
    junaid_encoding = fg.face_encodings(junaid_img)[0]

    known_encoding = [ usama_encoding , junaid_encoding]

    known_img_name = ["Muhammad Usama Aleem", "Junaid"]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #small_img = cv2.resize(unknown_img , (0 , 0) , fx=0.25 , fy=0.25)

    rgb_small_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #if process_this_img :
    unknown_img_location = fg.face_locations(rgb_small_img)
    unknown_img_encoding = fg.face_encodings(rgb_small_img , unknown_img_location)

    pil_image = Image.fromarray(img)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    for (top , left , bottom , right) , test_faces in zip(unknown_img_location , unknown_img_encoding) :
        name = "unknown"
        matches = fg.compare_faces(known_encoding , test_faces)
        if True in matches :
            first_match_index = matches.index(True)
            name = known_img_name[ first_match_index ]
            print("found it")
            print(name)
            name_face.append(name)
        draw.rectangle(((left , top) , (right , bottom)) , outline=(0 , 0 , 255))

        # Draw a label with a name below the face
        text_width , text_height = draw.textsize(name)
        draw.rectangle(((left , bottom + text_height + 10) , (right , bottom)) , fill=(0 , 0 , 255) ,
                       outline=(0 , 0 , 255))
        draw.text((left + 6 , bottom + text_height + 5) , name , fill=(255 , 255 , 255 , 255))

    del draw

    # Display the resulting image
    pil_image.show()


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
    face_recognition(img)
    speak(name_face)
speak(final)
cv2.waitKey(0)
cv2.destroyAllWindows()
