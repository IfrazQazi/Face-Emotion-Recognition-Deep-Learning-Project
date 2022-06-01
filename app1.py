# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:4:37 2022

@author:IFRAZ
"""
# extra
from bokeh.models.widgets import Div
# Importing required libraries
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,

)

# load model

emotion_dict = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# load json and create model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("fer.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoProcessorBase):
    # def transform(self, frame):
    #     img = frame.to_ndarray(format="bgr24")
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    
    st.sidebar.markdown("INVOLVED TEAM MEMBERS:")
    st.sidebar.markdown("IFRAZ QAZI:")
    st.sidebar.markdown("ifraz.qazi@somaiya.edu")
    if st.sidebar.button(
       """
        Ifraz LinkedIn  """ ):
        js = "window.open('https://www.linkedin.com/in/ifraz-kazi-01b8a721b/')"  # New tab or window
        js = "window.location.href = 'https://www.linkedin.com/in/ifraz-kazi-01b8a721b/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)
        

    st.sidebar.markdown("AKRAM KHAN:")

    st.sidebar.markdown("akramkhan0799@gmail.com")
    if st.sidebar.button(
       """
        Akram LinkedIn  """ ):
        js = "window.open('https://www.linkedin.com/in/akram-khan-300704220/')"  # New tab or window
        js = "window.location.href = 'https://www.linkedin.com/in/akram-khan-300704220/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

    
    st.sidebar.markdown("SHAILENDRA DUBEY:")

    st.sidebar.markdown("shailendradubey114@gmail.com")
    if st.sidebar.button(
       """
        Shailendra LinkedIn  """ ):
        js = "window.open('https://www.linkedin.com/in/shailendra-d-96402912a/')"  # New tab or window
        js = "window.location.href = 'https://www.linkedin.com/in/shailendra-d-96402912a/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)
   
    st.sidebar.markdown("UMESH RATHOD:")

    st.sidebar.markdown("umesh.rathod1307@gmail.com")
    if st.sidebar.button(
       """
        Umesh LinkedIn  """ ):
        js = "window.open('https://www.linkedin.com/in/umesh-rathod-894720186/')"  # New tab or window
        js = "window.location.href = 'https://www.linkedin.com/in/umesh-rathod-894720186/'"  # Current tab
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)
   
            
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has functionality to
                
                 detect Real time face emotion recognization.
                 """)
        st.image('emoition_image.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Observant Force Group using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect,
                                  just click on the link in sidebar and shoot us a request. If you have any suggestion or want to comment just write a mail at saurabhyd423@gmail.com/shubhamdeshmukh278@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()