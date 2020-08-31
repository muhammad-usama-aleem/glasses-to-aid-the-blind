import cv2
import face_recognition as fg
from PIL import Image, ImageDraw

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