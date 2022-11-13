import streamlit as st
import numpy as np 
import pandas as pd 
import cv2
import os 
import mahotas
import pickle
from joblib import load

class document_liveness_detection:

    def predict_result(self,x_image_scaled):
        loaded_model = pickle.load(open("rf_model.pkl", 'rb'))
        result = loaded_model.predict(x_image_scaled)
        if result >0.5:
            return "Live"
        else:
            return "Fake"

    def save_frame(self,video_path):
        filename, extension = os.path.splitext(video_path)
        # Load in video capture
        cap = cv2.VideoCapture(video_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id= int(nframes/2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        #print(frame,ret)
        cv2.imwrite('frame.png', frame)
    
    def read_img(self,frame="frame.png"):
        x_image=[]
        image = cv2.imread(frame)
        # image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #image = cv2.flip(img, 0)
        image_resized = cv2.resize(image, dsize=(256,256),interpolation=cv2.INTER_AREA)
        cv2.imwrite("resized.png",image_resized)
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # HUMoments for shape
        image_hu = cv2.HuMoments(cv2.moments(image_gray)).flatten()
        # Haralick for texture  
        image_har = mahotas.features.haralick(image).mean(axis=0)
        # convert the image to HSV color-space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # color histogram
        image_hist  = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(image_hist, image_hist)
        image_hist_flat = image_hist.flatten()
        # combine the features extracted
        x_image = np.hstack([image_hist_flat, image_har, image_hu])
        return x_image

    def normalize_img(self,x_image):
        scaler = load("std_scaler.bin")
        x_image_scaled = scaler.transform([x_image])
        #print(x_image_scaled)
        return x_image_scaled

    def construct_app(self):
        st.header("Document Liveness Detection")
        st.markdown("Document Liveness Detection Bad actors present digital images of identification documents instead of “live” IDs in their possession to spoof identity verification processes and commit fraud. ")
        st.markdown("Document liveness detection uses AI and computer vision to distinguish between a present document and a digital image shown on a mobile device or computer display to prevent this type of presentation attack.")
        uploaded_file = st.file_uploader("Upload your video or image:")
        st.write("Live or Fake Document?")
        if uploaded_file:
            with open(uploaded_file.name,"wb") as f:
                f.write(uploaded_file.getbuffer())    
            self.save_frame(uploaded_file.name)
            x_image = self.read_img()
            x_image_scaled = self.normalize_img(x_image)
            prediction_str= self.predict_result(x_image_scaled)
            st.write(f"{prediction_str}")
            

a = document_liveness_detection()
a.construct_app()