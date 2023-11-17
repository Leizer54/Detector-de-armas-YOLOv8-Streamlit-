import cv2
import streamlit as st
import tempfile
from ultralytics import YOLO

def DetectVideo():

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

