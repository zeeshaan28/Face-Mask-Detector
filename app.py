import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import av
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title ="Face Mask Detector", page_icon='🦈' , layout='centered', initial_sidebar_state='auto')


def mask_image():
    global RGB_img
    # load our serialized face detector model
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from local and grab the image spatial dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




def webcam():
    
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])

    class VideoTransformer(VideoTransformerBase):
    
        
        
        def __init__(self)-> None:
            self._net = cv2.dnn.readNet(prototxtPath, weightsPath)
            self._model =(load_model("mask_detector.model"))
            self._confidence_threshold = 0.5
            
           
        def _annotate_image(self, image, detections):
            
            faces = []
            locs = []
            preds = []
                    # loop over the detections
            (h, w) = image.shape[:2]
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
        
                if confidence > self._confidence_threshold:
                    
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
        
                            # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                            # extract the face ROI, convert it from BGR to RGB channel
                            # ordering, resize it to 224x224, and preprocess it
                    face = image[startY:endY, startX:endX]
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    
                    
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
                    
                    
            if len(faces)>0:
                    
                faces = np.array(faces, dtype="float32")
                preds= self._model.predict(faces, batch_size=32)
                
                
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask,withoutMask) = pred


        		# determine the class label and color we'll use to draw
    		    # the bounding box and text	
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                
                # display the label and bounding box rectangle on the output
                # frame
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                            
            return image      
        
        def transform(self, frame: av.VideoFrame) -> np.ndarray:
                    image = frame.to_ndarray(format="bgr24")
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
                    self._net.setInput(blob)
                    detections = self._net.forward()
                    output_img=self._annotate_image(image,detections)
                    
                    return output_img
                
    webrtc_streamer(key="example",video_transformer_factory=VideoTransformer)


def face_mask_detection():
    st.markdown('<h1 align="center">Face Mask Detection 👦 </h1>', unsafe_allow_html=True)
    activities = [None,"Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection using?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image 📷</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg","gif","jfif"])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                mask_image()
                st.image(RGB_img, use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam 📹 </h2>', unsafe_allow_html=True)
        st.markdown("### Click the start button for capturing")
        #if st.button('Open Webcam'):
        webcam()
            
    if choice== None:
        st.write('')
        st.write(' This project is developed considering the amid second wave of coronavirus in india. Second wave has hit the country very badly, The number of active cases are increasing rapidly and death rate has also increased.')
        st.write(' Viewing this situation, It has become very important to take precautions like wearing face mask, frequent sanitization and social distancing to protect ourselves and our loved ones from this lethal virus. Health Organsations, government, news channels, everybody is stressing on wearing a mask whenever going out. This is the first important thing we have to do to stop the virus chain and prevent it from harming us.') 
        st.write('So, This project aims at detecting the people not wearing a mask. A Face Mask Classifier model (ResNet50) is trained and deployed for accurate detection. For aiding the training process, augmented masked faces are generated (using facial landmarks) and blurring effects are also imitated.')
        st.markdown('<h3 align="center"> For detection, Please select any of the option from sidebar</h3>', unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.text('This project is under furthur development, working on the machine learning model which can detect the violation of social distancing.')
        st.text('If interested, can check my github profile for furthur updates.')
        link = '[GitHub](https://github.com/zeeshaan28/Face-Mask-Detector)'
        st.markdown(link, unsafe_allow_html=True)
            
face_mask_detection()