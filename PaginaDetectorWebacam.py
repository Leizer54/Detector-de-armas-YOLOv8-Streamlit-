import streamlit as st
import cv2
from ultralytics import YOLO

#import smtplib
#from email.mime.multipart import MIMEMultipart
#from email.mime.text import MIMEText

def DetectWebcam():

    # Modelo YOLO
    model_path = "weight/best.pt"

    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    with st.sidebar:
        st.markdown('---')
        confidence = float(st.slider(
            "Confianza del modelo", 0, 100, 38)) / 100
        st.markdown('---')

    st.title("Detección en Webcam")

    if st.sidebar.button("Realizar detección"):
        try:
            flag = st.button(
                label="Stop running"
            )
            vid_cap = cv2.VideoCapture(0)  # local camera
            st_frame = st.empty()
            while not flag:
                success, image = vid_cap.read()     # Captura un fotograma de la webcam y lo guarda en image. success indica si la captura fue exitosa.
                if success:
                    image = cv2.resize(image, (720, int(720 * (9 / 16))))
                    res = model.predict(image, conf=confidence)     # Se predice con el modelo seleccionado y se guarda en res
                    res_plotted = res[0].plot()         # Crea una visualización con los objetos detectados y se guarda en res_plotted
                    st_frame.image(res_plotted,         # Muestra la visualización de detección dentro de st_frame
                                   caption='Detección en Webcam',
                                   channels="BGR",
                                   use_column_width=True
                                   )

                else:
                    vid_cap.release()
                    break

        except Exception as e:
            st.error(f"Error loading video: {str(e)}")


