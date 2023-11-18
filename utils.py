import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

def DetectImage():

    # Modelo YOLO
    model_path = "weight/best.pt"

    with st.sidebar:
        st.markdown('---')
        confidence = float(st.slider(
            "Confianza del modelo", 0, 100, 38)) / 100
        st.markdown('---')
        source_img = st.file_uploader(
            "Seleccione una imagen", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    st.title("Detección en imagen")
    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = PIL.Image.open(source_img)  # Opening the uploaded image
            st.image(source_img, caption="Imagen cargada",
                     use_column_width=True
                     )
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    if st.sidebar.button('Realizar detección'):
        res = model.predict(uploaded_image,
                            conf=confidence
                            )
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                     caption='Detección',
                     use_column_width=True
                     )
        st.write("---")
        st.markdown("*Creado por Leandro Leizer Berrospi Perez*")


def DetectVideo():

    global model
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
        source_vid = st.file_uploader(
            "Seleccione un video", type=("mp4","mov","avi",'asf','m4v'))

    st.title("Detección en video")

    if source_vid:
        st.sidebar.video(source_vid)

        if st.sidebar.button("Realizar detección"):
            with st.spinner("Detectando..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_vid.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            image = cv2.resize(image, (720, int(720 * (9 / 16))))

                            # Predict the objects in the image using YOLOv8 model
                            res = model.predict(image, conf=confidence)

                            # Plot the detected objects on the video frame
                            res_plotted = res[0].plot()
                            st_frame.image(res_plotted,
                                           caption='Video detectado',
                                           channels="BGR",
                                           use_column_width=True
                                           )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")
        st.write("---")
        st.markdown("*Creado por Leandro Leizer Berrospi Perez*")

def DetectWebcam():

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

