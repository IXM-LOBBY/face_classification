from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from pythonosc import osc_message_builder
from pythonosc import udp_client

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Setup OSC Client
client = udp_client.SimpleUDPClient("127.0.0.1", 9010)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
# seems to be necessary for some cameras - grabbing a couple frames in the beginning
video_capture.read()
video_capture.read()

# capture and process loop
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    emotionRange = ['','','','','','','']
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        #0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        #print (emotion_prediction)
        #print ("Angry: ", round(emotion_prediction[0][0], 3))
        #print ("Disgust: ", round(emotion_prediction[0][1], 3))
        #print ("Fear: ", round(emotion_prediction[0][2], 3))
        #print ("Happy: " + str(round(emotion_prediction[0][3], 3)))
        
        emotionRange[0] = "Angry: " + str(round(emotion_prediction[0][0], 3))
        emotionRange[1] = "Disgust: " + str(round(emotion_prediction[0][1], 3))
        emotionRange[2] = "Fear: " + str(round(emotion_prediction[0][2], 3))
        emotionRange[3] = "Happy: " + str(round(emotion_prediction[0][3], 3))
        emotionRange[4] = "Sad: " + str(round(emotion_prediction[0][4], 3))
        emotionRange[5] = "Surprise: " + str(round(emotion_prediction[0][5], 3))
        emotionRange[6] = "Neutral: " + str(round(emotion_prediction[0][6], 3))
        
        # Create / Send OSC Message
        msg = osc_message_builder.OscMessageBuilder(address="/emotion/scores")
        msg.add_arg(float(round(emotion_prediction[0][0], 3)))
        msg.add_arg(float(round(emotion_prediction[0][1], 3)))
        msg.add_arg(float(round(emotion_prediction[0][2], 3)))
        msg.add_arg(float(round(emotion_prediction[0][3], 3)))
        msg.add_arg(float(round(emotion_prediction[0][4], 3)))
        msg.add_arg(float(round(emotion_prediction[0][5], 3)))
        msg.add_arg(float(round(emotion_prediction[0][6], 3)))
        client.send(msg.build())
        
        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(bgr_image,emotionRange[0],(0,30), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[1],(0,60), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[2],(0,90), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[3],(0,120), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[4],(0,150), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[5],(0,180), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(bgr_image,emotionRange[6],(0,210), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
