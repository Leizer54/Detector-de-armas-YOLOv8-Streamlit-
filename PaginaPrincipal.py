import requests
import streamlit as st
from streamlit_lottie import st_lottie
import PaginaDetectorImage, PaginaDetectorVideo, PaginaDetectorWebacam

# Configuración de la página
st.set_page_config(
    page_title="Detector de armas",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Animación de Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Interfaz de la página "Home"
def main():

    # Barra lateral
    st.sidebar.image(
        "https://github.com/Leizer54/Detector-de-armas-YOLOv8-Streamlit-/blob/main/OIG.LtUmN-removebg-preview.png?raw=true")

    app_mode = st.sidebar.selectbox('¿Qué desea hacer?',
                                    ['Seleccionar...', 'Detectar en imagen', 'Detectar en video','Detectar en Webcam'])

    if app_mode == 'Seleccionar...':
        st.title("Detector de armas")
        st.markdown("Bienvenido a la aplicación de detecctión de armas de la Universidad Tecnológica del Perú.")

        with st.container():
            left_column, right_column = st.columns(2)
            with left_column:
                st.header("Introducción")
                st.write("La detección de armas en tiempo real es crucial para fortalecer "
                    "la seguridad en diversos entornos, ya que proporciona una respuesta"
                    " inmediata ante amenazas potenciales. En entornos públicos, instalaciones"
                    " críticas y eventos masivos, esta tecnología contribuye a prevenir crímenes"
                    " violentos y actos terroristas, mejorando la capacidad de reacción de las "
                    "fuerzas de seguridad. Además, la detección temprana de armas facilita la "
                    "identificación rápida de situaciones de emergencia, lo que es fundamental "
                    "para proteger vidas y propiedades.")
            with right_column:
                lottie_hello = load_lottieurl("https://lottie.host/771ea30c-dbce-446d-87ee-510b56927b53/Hly7dpw9w2.json")
                st_lottie(lottie_hello,
                  speed=1,
                  height=400,
                  width=400,
                  key="hello")

        with st.container():
            left_column1, right_column2 = st.columns(2)
            with left_column1:
                lottie_detect = load_lottieurl("https://lottie.host/aecab677-46a6-4899-9b79-ec778664be9c/QIzrqTCXd9.json")
                st_lottie(lottie_detect,
                          speed=0.5,
                          height=350,
                          width=350,
                          key="detect")
            with right_column2:
                st.header("El sistema")
                st.write("Este sistema usa el modelo de detección de objetos "
                         "en tiempo real y segmentación de imágenes YOLOv8, el cual se basa en avances "
                         "de vanguardia en aprendizaje profundo y visión por computadora, "
                         "ofreciendo un rendimiento incomparable en términos de velocidad y precisión.")

                st.write("En la barra lateral izquierda puede encontrar varias opciones "
                         "donde se puede ejecutar el sistema. Desde imágenes y videos "
                         "hasta detección en tiempo real."
                         " Suba una imagen, un video o active la cámara"
                         " y haga click en *Ejecutar* para iniciar la detección.")
        st.write("---")
        st.markdown("*Creado por Leandro Leizer Berrospi Perez*")

    # Interfaz para cuando se quiera "Detectar en IMAGEN"
    elif app_mode == 'Detectar en imagen':
        PaginaDetectorImage.DetectImage()

    # Interfaz para cuando se quiera "Detectar en VIDEO"
    elif app_mode == 'Detectar en video':
        PaginaDetectorVideo.DetectVideo()

    # Interfaz para cuando se quiera "Detectar en WEBCAM"
    elif app_mode == 'Detectar en Webcam':
        PaginaDetectorWebacam.DetectWebcam()



if __name__== '__main__':
    try:
        main()
    except SystemExit:
        pass