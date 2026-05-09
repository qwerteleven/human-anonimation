# 🕵️ Human Anonymization — Face & Person Blurring Pipeline

Sistema de **anonimización automática de personas** en imágenes mediante detección de rostros y cuerpos completos, aplicando desenfoque gaussiano sobre las regiones detectadas. Diseñado para proteger la privacidad en imágenes de videovigilancia, datasets públicos y cualquier contexto donde sea necesario cumplir con normativas como el [RGPD](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation).

---

## ¿Qué problema resuelve?

Publicar imágenes con personas identificables plantea serios problemas legales y éticos. La [anonimización de datos](https://en.wikipedia.org/wiki/Data_anonymization) mediante técnicas de visión computacional permite procesar grandes volúmenes de imágenes automáticamente, sin intervención manual, garantizando que ningún rostro ni cuerpo sea reconocible en el resultado final.

Este repositorio combina dos enfoques complementarios:

- **Detección de personas** (cuerpo completo) mediante un modelo semi-supervisado.
- **Detección facial** de alta precisión con RetinaFace.

---

## Arquitectura del sistema

### 1. Detección de personas — SoftTeacher (Semi-Supervised Object Detection)

El módulo principal (`ssod`) está construido sobre [SoftTeacher](https://arxiv.org/abs/2106.09018), un framework de [detección de objetos semi-supervisada](https://en.wikipedia.org/wiki/Semi-supervised_learning). Utiliza un esquema **teacher-student** donde:

- El modelo *teacher* genera pseudo-etiquetas sobre imágenes no anotadas.
- El modelo *student* aprende de esas pseudo-etiquetas junto con los datos etiquetados reales.

Esto permite entrenar detectores robustos con muy pocas anotaciones manuales, ideal para datasets de personas en entornos variados (niebla, lluvia, noche, tormenta de arena).

La inferencia usa [MMDetection](https://github.com/open-mmlab/mmdetection) como backend, un framework de detección estándar en investigación basado en [PyTorch](https://en.wikipedia.org/wiki/PyTorch).

### 2. Detección facial — RetinaFace

[RetinaFace](https://arxiv.org/abs/1905.00641) es un detector facial del estado del arte que localiza rostros con gran precisión incluso en condiciones adversas (oclusión, escala reducida, ángulos extremos). Se integra a través de la librería [InsightFace](https://github.com/deepinsight/insightface).

```python
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → GPU (mínimo 8 GB VRAM)
faces = app.get(image)
```

### 3. Anonimización — Gaussian Blur por máscara

Una vez detectadas las regiones de interés, se aplica [desenfoque gaussiano](https://en.wikipedia.org/wiki/Gaussian_blur) sobre un fondo difuminado usando OpenCV:

```python
blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
mask = np.zeros(img.shape, dtype=np.uint8)
# Solo se desenfoca dentro de la bounding box detectada
out = np.where(masked != [255,255,255], img, blurred_img)
```

El kernel de desenfoque se escala proporcionalmente al tamaño de la imagen (`factor=3`), garantizando que la intensidad del blur sea consistente independientemente de la resolución.

---

## Ejemplos

### Ground Truth vs. Inferencia

| Original | Anonimizado |
|----------|-------------|
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/FIXLIUCA4BHXPMVKCWHZFESN5M-scaled.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/blured_img1.jpg" width="370"> |

### Más ejemplos — Condiciones adversas

| | |
|--|--|
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/beach_.jpeg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/fog_20.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/night_12.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/night_14.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/fog_4.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_10.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_7.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_1.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_19.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_4.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/snow_15.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_14.jpg" width="370"> |

El sistema mantiene un buen rendimiento en escenas de playa, niebla, lluvia, nieve y tormenta de arena, condiciones donde los detectores clásicos suelen degradarse.

---

## Instalación

**Requisitos:** Python 3.6 – 3.8, CUDA (recomendado, mínimo 8 GB VRAM para RetinaFace en GPU)

```bash
# 1. Clonar el repositorio
git clone https://github.com/qwerteleven/human-anonimation.git
cd human-anonimation

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Instalar el paquete ssod
pip install -e .
```

Dependencias principales:

| Librería | Uso |
|----------|-----|
| `torch` / `torchvision` | Motor de deep learning ([PyTorch](https://en.wikipedia.org/wiki/PyTorch)) |
| `mmcv-full` | Utilidades de visión computacional (OpenMMLab) |
| `insightface` | Detección facial con RetinaFace |
| `wandb` | Tracking de experimentos ([Weights & Biases](https://wandb.ai)) |
| `prettytable` | Visualización de métricas en terminal |

---

## Uso

### Anonimización de imágenes (modo RetinaFace)

```bash
python demo/image_demo.py "path/to/images/*.jpg" config.py checkpoint.pth \
    --device cuda:0 \
    --score-thr 0.5 \
    --output output_blur/
```

Los resultados se guardan en:
- `retinaface/output/` — imagen con bounding boxes visualizadas
- `retinaface/output_blur/` — imagen con rostros desenfocados

### Anonimización en modo asíncrono

```bash
python demo/image_demo.py "path/to/images/*.jpg" config.py checkpoint.pth \
    --async-test \
    --output output_blur/
```

El modo asíncrono permite procesar múltiples imágenes en paralelo usando `asyncio`, reduciendo el tiempo total en lotes grandes.

### Parámetros clave

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--score-thr` | Umbral de confianza mínimo para aceptar una detección | `0.5` |
| `--device` | Dispositivo de inferencia (`cuda:0`, `cpu`) | `cuda:0` |
| `factor` | Intensidad del desenfoque (menor = más difuso) | `3` |
| `det_size` | Resolución de entrada para RetinaFace | `(640, 640)` |

---

## Estructura del proyecto

```
human-anonimation/
├── demo/
│   └── image_demo.py        # Script principal: detección + anonimización
├── ssod/                    # Módulo Semi-Supervised Object Detection
│   ├── apis/
│   │   └── inference.py     # init_detector, save_result
│   └── utils/               # patch_config y utilidades
├── requirements.txt
└── setup.py                 # Instalación del paquete ssod
```

---

## Pipeline completo

```
Imagen de entrada
      │
      ▼
┌─────────────────────┐     ┌──────────────────────┐
│   RetinaFace        │     │   SoftTeacher (SSOD)  │
│   (detección        │     │   (detección de        │
│    facial)          │     │    personas completas) │
└────────┬────────────┘     └──────────┬─────────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
         Bounding boxes detectadas
         (score > 0.7 para blur)
                    │
                    ▼
         Gaussian Blur por máscara
                    │
                    ▼
         Imagen anonimizada guardada
```

---

## Contexto legal y ético

Este sistema está diseñado para facilitar el cumplimiento de normativas de privacidad como el [RGPD (Reglamento General de Protección de Datos)](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) europeo, que considera los datos biométricos (incluyendo rostros) como datos de categoría especial sujetos a protección reforzada.

La anonimización automática permite publicar datasets de imágenes en espacios públicos, procesar grabaciones de cámaras de seguridad y construir sistemas de análisis sin comprometer la identidad de las personas.

---

## Trabajo relacionado y referencias

| Concepto | Referencia |
|----------|------------|
| Aprendizaje semi-supervisado | [Semi-supervised learning — Wikipedia](https://en.wikipedia.org/wiki/Semi-supervised_learning) |
| Detección de objetos | [Object detection — Wikipedia](https://en.wikipedia.org/wiki/Object_detection) |
| Redes neuronales convolucionales | [Convolutional neural network — Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) |
| Desenfoque gaussiano | [Gaussian blur — Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur) |
| Anonimización de datos | [Data anonymization — Wikipedia](https://en.wikipedia.org/wiki/Data_anonymization) |
| RGPD / privacidad | [General Data Protection Regulation — Wikipedia](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) |
| Bounding box | [Bounding box — Wikipedia](https://en.wikipedia.org/wiki/Bounding_box) |
| PyTorch | [PyTorch — Wikipedia](https://en.wikipedia.org/wiki/PyTorch) |