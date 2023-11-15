# Detector-de-armas-YOLOv8-Streamlit-
Aplicación web de detección de armas en tiempo real mediante YOLOv8n

##  Requerimientos:
### Creación de un entorno virtual y activación:
```python
pip install virtual venv
```
```python
virtual venv
```
```python
source venv/bin/activate
```

### Instalación de Torch y Torchvision habilitados para CUDA
- Ir al sitio web oficial de PyTorch (https://pytorch.org/)
- Seleccionar las opciones de instalación adecuadas.
- Copiar el comando de instalación.
- Pegar el comando de instalación en el terminal

Para este trabajo se usó CUDA 12.1 Python en Windows:

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Eso instala automáticamente Torchvision
- Comprobar que Pytorch y Torchvision funcionan:
```python
import torch
import torchvision
print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)
```

### Instalación  de Ultralytics, Streamly y Pafy
```python
pip install ultralytics, streamlit, pafy
```
> Se debe tener descargados los pesos de YOLOv8 ya entrenados para poder usarse en este proyecto. Crear un directorio con dichos pesos
