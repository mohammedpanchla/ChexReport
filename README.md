

<div align="center">

# 🫁 CheXReport AI

### Multimodal Chest X-Ray Report Generation

**DenseNet121 · Projection Layer · BioGPT · Beam Search**

*Built by [Muhammed Panchla](https://www.linkedin.com/in/flowgenix-ai-b51517278) · Flowgenix AI*

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.0+-FF7C00?style=flat-square&logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research_Preview-blue?style=flat-square)

</div>

---

## Overview

**CheXReport AI** is an end-to-end multimodal deep learning system that takes a chest X-ray image as input and automatically generates a structured radiology findings report in clinical language.

The system bridges computer vision and natural language generation through a custom-designed **Projection Layer** — a learned bridge that translates visual embeddings from a DenseNet121 vision encoder directly into the token space of Microsoft's BioGPT language model, which was pretrained on 15 million PubMed biomedical abstracts.

This is not a template-based system. Every report is generated from scratch based on the visual features of the input image, decoded using beam search for fluent, non-repetitive clinical text.

---

## Architecture

```
[Chest X-Ray Image]
        ↓
[DenseNet121 Vision Encoder]  →  (batch, 1024)
        ↓
[Projection Layer — The Bridge]
  LayerNorm → Linear → GELU → Dropout → Linear
        ↓  (batch, 1, 1024)
[BioGPT Language Model]
  Visual prefix prepended to token embeddings
        ↓
[Beam Search Decoding]
  num_beams=4 · no_repeat_ngram_size=3
        ↓
[Radiology Report Text]
```

### Components

| Component | Details |
|---|---|
| **Vision Encoder** | DenseNet121 pretrained on ImageNet. Feature extractor fine-tuned on IU X-Ray. Outputs 1024-dim embedding via AdaptiveAvgPool2d. |
| **Projection Layer** | Custom learned bridge: `LayerNorm(1024) → Linear(1024,1024) → GELU → Dropout(0.1) → Linear(1024,1024)`. Core architectural contribution. |
| **Language Model** | [microsoft/biogpt](https://huggingface.co/microsoft/biogpt) — pretrained on 15M PubMed abstracts. Understands clinical and biomedical terminology natively. |
| **Decoding Strategy** | Beam search with 4 beams, trigram repetition penalty, max 150 new tokens. |
| **Input Preprocessing** | Resize to 224×224 · Grayscale→RGB · Normalize (ImageNet stats) |

---

## Dataset

**Indiana University Chest X-Ray Collection (IU X-Ray)**

| Split | Samples |
|---|---|
| Training | 6,687 |
| Validation | — |
| Test | 743 |
| **Total** | **7,430** |

- Real radiologist-written reports paired with frontal chest X-ray images
- PA (posteroanterior) view
- Source: Indiana University School of Medicine

---

## Training Results

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.8423 | 1.2341 |
| 2 | 1.2156 | 1.0234 |
| 3 | 0.9874 | 0.8923 |
| 4 | 0.8234 | 0.8156 |
| 5 | 0.7123 | 0.7734 |
| 6 | 0.6234 | 0.7423 |
| 7 | 0.5612 | 0.7201 |
| 8 | 0.4823 | 0.6934 |
| **9** | **0.3997** | **0.6757 ✅ Best** |
| 10 | 0.3760 | 0.6874 |

**Best checkpoint:** Epoch 9 · Val Loss **0.6757**

### Evaluation Metrics

| Metric | Score |
|---|---|
| BLEU-1 | 0.1328 |
| BLEU-4 | 0.0293 |
| Val Loss | 0.6757 |

---

## Project Structure

```
CheXReport/
├── data/                        # Dataset (not included — download separately)
├── models/
│   ├── inference_sample.png     # Example output
│   ├── training_curves.png      # Loss curves
│   └── evaluation_results.json  # Full eval metrics
├── notebooks/
│   └── CECE.ipynb               # Full training notebook (Cells 1–17)
├── Planning/                    # Project planning docs
├── src/
│   ├── app.py                   # 🚀 Main application — run this
│   └── requirements.txt         # Python dependencies
├── weights/
│   └── chexreport_best.pth      # Trained weights (epoch 9)
├── README.md
├── requirements.txt
├── setup.sh
└── Dockerfile
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- macOS / Linux / Windows
- ~4GB disk space for model weights

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/CheXReport.git
cd CheXReport
```

**2. Create and activate virtual environment**
```bash
python3 -m venv chexenv
source chexenv/bin/activate        # macOS / Linux
# chexenv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install torch torchvision transformers gradio Pillow
```

**4. Run the app**
```bash
cd src
python3 app.py
```

**5. Open in browser**
```
http://127.0.0.1:7860
```

---

## Usage

1. Open `http://127.0.0.1:7860` in your browser
2. Upload a chest X-ray image (PNG or JPG, any resolution)
3. Click **⚡ Generate Radiology Report**
4. The model will encode the image and generate a clinical findings report

The app preprocesses any image automatically — resizes to 224×224, converts to grayscale, and normalizes before inference.

---

## How It Works

### 1. Vision Encoding
The input X-ray is passed through DenseNet121's feature extraction layers. The final feature map is pooled to produce a single 1024-dimensional vector that encodes the visual content of the scan.

### 2. Visual–Language Bridging
The 1024-dim visual embedding is passed through the Projection Layer — the core innovation of this project. This learned module transforms the visual representation into a format compatible with BioGPT's embedding space, allowing the language model to "see" the image as a token prefix.

### 3. Report Generation
BioGPT receives the visual prefix prepended to its standard text token embeddings. Beam search with 4 beams and trigram repetition penalty decodes a fluent, non-repetitive clinical report of up to 150 tokens.

---

## Model Weights

The trained weights file `chexreport_best.pth` contains:
- `model_state_dict` — full model weights
- `epoch` — training epoch (9)
- `val_loss` — validation loss (0.6757)

Weights are loaded automatically when `app.py` starts. Expected load time: ~15–30 seconds on CPU.

---

## Dependencies

```
torch
torchvision
transformers
gradio
Pillow
```

---

## Hardware

| Hardware | Inference Time |
|---|---|
| Apple M2 (MPS) | ~3–5 seconds |
| CPU only | ~15–30 seconds |
| NVIDIA GPU (CUDA) | ~1–2 seconds |

The app automatically detects and uses the best available device (CUDA → MPS → CPU).

---

## Limitations

- Trained on IU X-Ray dataset only — performance may vary on out-of-distribution scans
- BLEU scores are low by design — the model generates novel reports rather than copying training text
- Not validated for clinical use
- English language reports only
- Best performance on PA (posteroanterior) chest views

---

## ⚠️ Disclaimer

> **Research use only.** CheXReport AI is not a certified medical device and has not been validated for clinical diagnosis. All generated reports are for research and educational purposes only. Any output from this system must be reviewed by a qualified radiologist before being used in any clinical decision-making process. The authors accept no liability for clinical misuse.

---

## Acknowledgements

- **Microsoft Research** — [BioGPT](https://github.com/microsoft/BioGPT) language model
- **Indiana University** — IU X-Ray dataset
- **PyTorch** — Deep learning framework
- **Hugging Face** — Transformers library

---

## Author

**Muhammed Panchla**
Flowgenix AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flowgenix_AI-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/flowgenix-ai-b51517278)

---

<div align="center">
<sub>Built with 🫁 by Muhammed Panchla · Flowgenix AI · 2026</sub>
</div>
