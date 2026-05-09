# 🕵️ Human Anonymization — Face & Person Blurring Pipeline

Automatic anonymization system for people in images using face and full-body detection, applying Gaussian blurring to the detected regions. Designed to protect privacy in video surveillance images, public datasets, and any context where compliance with regulations such as the GDPR is required.

---

## What problem does it solve?

Publishing images with identifiable people raises serious legal and ethical issues. Data anonymization using computer vision techniques allows for the automatic processing of large volumes of images, without manual intervention, ensuring that no face or body is recognizable in the final result.

This repository combines two complementary approaches:

- **Person detection** (full body) using a semi-supervised model.

- High-accuracy **facial detection** with RetinaFace.

---

## System Architecture

### 1. Person Detection — SoftTeacher (Semi-Supervised Object Detection)

The main module (`ssod`) is built on SoftTeacher (https://arxiv.org/abs/2106.09018), a semi-supervised object detection framework (https://en.wikipedia.org/wiki/Semi-supervised_learning). It uses a teacher-student approach where:

- The teacher model generates pseudo-labels on unannotated images.

- The student model learns from these pseudo-labels along with the actual labeled data.

This allows for the training of robust detectors with very little manual annotation, ideal for datasets of people in varied environments (fog, rain, night, sandstorm).

The inference uses MMDetection as the backend, a research-standard detection framework based on PyTorch.

### 2. Facial Detection — RetinaFace

RetinaFace is a state-of-the-art facial detector that locates faces with high accuracy even under adverse conditions (occlusion, reduced scale, extreme angles). It is integrated through the InsightFace library.

```python
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 → GPU (minimum 8 GB VRAM)
faces = app.get(image)
```

### 3. Anonymization — Gaussian Blur by Mask

Once the regions of interest are detected, Gaussian blur is applied to a blurred background using OpenCV:

```python
blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
mask = np.zeros(img.shape, dtype=np.uint8)
# Blurring only occurs within the detected bounding box
out = np.where(masked != [255,255,255], img, blurred_img)
```

The blur kernel scales proportionally to the image size (`factor=3`), ensuring consistent blur intensity regardless of resolution.

---

## Examples

### Ground Truth vs. Inference

| Original | Anonymized |

|----------|-------------|

<img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/FIXLIUCA4BHXPMVKCWHZFESN5M-scaled.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/blured_img1.jpg" width="370"> |

### More examples — Adverse conditions

| | |
|--|--|
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/beach_.jpeg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/fog_20.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/night_12.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonymation/blob/main/assets/night_14.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/fog_4.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_10.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_7.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_1.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_19.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/rain_4.jpg" width="370"> |
| <img src="https://github.com/qwerteleven/human-anonymation/blob/main/assets/snow_15.jpg" width="370"> | <img src="https://github.com/qwerteleven/human-anonimation/blob/main/assets/sandstorm_14.jpg" width="370"> |

The system maintains good performance in beach, fog, rain, snow and sandstorm scenes, conditions where classic detectors tend to degrade.

---

## Installation

**Requirements:** Python 3.6 – 3.8, CUDA ((Recommended, minimum 8 GB VRAM for RetinaFace on GPU)

```bash
# 1. Clone the repository
git clone https://github.com/qwerteleven/human-anonimation.git
cd human-anonimation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the ssod package
pip install -e .

```

Main dependencies:

| Library | Usage |

|----------|-----|

`torch` / `torchvision` | Deep learning engine ([PyTorch](https://en.wikipedia.org/wiki/PyTorch)) |

`mmcv-full` | Computer vision utilities (OpenMMLab) |

`insightface` | Facial detection with RetinaFace |

`wandb` | Experiment Tracking (Weights & Biases) |

`prettytable` | Terminal Metrics Visualization |


---

## Usage

### Image Anonymization (RetinaFace Mode)

```bash
python demo/image_demo.py "path/to/images/*.jpg" config.py checkpoint.pth \
--device cuda:0 \
--score-thr 0.5 \
--output output_blur/
```

The results are saved in:
- `retinaface/output/` — image with bounding boxes displayed
- `retinaface/output_blur/` — image with blurred faces

### Asynchronous Anonymization

```bash
python demo/image_demo.py "path/to/images/*.jpg" config.py checkpoint.pth \
--async-test \
--output output_blur/
```

Asynchronous mode allows processing multiple images in parallel using `asyncio`, reducing the total time for large batches.

### Key Parameters

| Parameter | Description | Default |

|-----------|-------------|---------|

`--score-thr` | Minimum confidence threshold to accept a detection | `0.5` |

`--device` | Inference device (`cuda:0`, `cpu`) | `cuda:0` |

`factor` | Blur intensity (lower = more blurred) | `3` |

`det_size` | Input resolution for RetinaFace | `(640, 640)` |

---

## Project Structure

```
human-anonymization/
├── demo/
│ └── image_demo.py # Main script: detection + anonymization
├── ssod/ # Semi-Supervised Object Detection Module
│ ├── apis/
│ │ └── inference.py # init_detector, save_result
│ └── utils/ # patch_config and utilities
├── requirements.txt
└── setup.py # Installing the ssod package
```

---

## Complete Pipeline

```
Input Image
│
▼
┌─────────────────────┐ ┌──────────────────────┐
│ RetinaFace │ │ SoftTeacher (SSOD) │
│ (facial detection) │ │ (full person detection)
│
│ │
└────────┬────────────┘ └──────────┬────────────┘

│ │

└──────────┬──────────────────┘
▼
Bounding boxes detected

(score > 0.7 for blur)

│

▼
Gaussian Blur by mask

│

▼
Anonymized image saved
```

---

## Legal and Ethical Context

This system is designed to facilitate compliance with privacy regulations such as the European GDPR (General Data Protection Regulation), which considers biometric data (including faces) as category data Special cases involving individuals requiring enhanced protection.

Automatic anonymization allows for the publication of image datasets in public spaces, the processing of security camera recordings, and the development of analytical systems without compromising people's identities.

---

## Related work and references

| Concept | Reference |

----------|------------|

Semi-supervised learning | [Semi-supervised learning — Wikipedia](https://en.wikipedia.org/wiki/Semi-supervised_learning) |

Object detection | [Object detection — Wikipedia](https://en.wikipedia.org/wiki/Object_detection) |

Convolutional neural networks | [Convolutional neural network — Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) |

Gaussian blur | [Gaussian blur — Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur) |

Data anonymization | [Data anonymization — Wikipedia](https://en.wikipedia.org/wiki/Data_anonymization) |
| GDPR / privacy | [General Data Protection Regulation — Wikipedia](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) |
| Bounding box | [Bounding box — Wikipedia](https://en.wikipedia.org/wiki/Bounding_box) |
| PyTorch | [PyTorch — Wikipedia](https://en.wikipedia.org/wiki/PyTorch) |
