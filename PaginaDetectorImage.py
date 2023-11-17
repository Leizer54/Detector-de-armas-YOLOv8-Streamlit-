import PIL
import streamlit as st
from ultralytics import YOLO

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
        #st.write("Model loaded successfully!")

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

        #    try:
        #        with st.expander("Resultados de la detección"):
        #            for box in boxes:
        #                st.write(box.xywh)
        #    except Exception as ex:
        #        st.write("No image is uploaded yet!")
