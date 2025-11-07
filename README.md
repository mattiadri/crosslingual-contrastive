# Multilingual CLIP Alignment via Image Pivoting

## Overview
This project investigates **multilingual multimodal alignment** between text and images using an **image as a semantic pivot**.  
The goal is to map captions in different languages into a shared embedding space **without using parallel text** during training.

The approach follows the **CLIP paradigm** (Contrastive Language–Image Pretraining), where the **visual encoder acts as a language-agnostic anchor**, encouraging semantically similar captions across languages to converge in the same space.  
This setup tests whether contrastive supervision from image–text pairs alone can induce meaningful **cross-lingual alignment**.

---

## Goals and Hypothesis
- **Goal**: fine-tune a CLIP-style model with multilingual image–caption pairs while avoiding any text-parallel supervision.  
- **Hypothesis**: the image pivot is sufficient to align language-specific text encoders into a shared semantic space.  
- **Focus**: representation analysis — how alignment evolves across languages and layers, before and after fine-tuning.

---

## Model Architecture

### Backbone
- **Vision Encoder**: OpenCLIP (e.g., ViT-B/32 pretrained on OpenAI or LAION)  
- **Text Encoder**: Multilingual CLIP-compatible SentenceTransformer (`clip-ViT-B-32-multilingual-v1`)  
- **Projection Head**: trainable linear mapping (ΔW) aligning multilingual text embeddings to the image space  

### Training Objective
- **Contrastive loss**: Symmetric InfoNCE on image–text pairs with in-batch negatives  
- **Regularization terms**:
  - *Prox-ID*: penalizes deviation from identity (‖W − I‖²) for stability  
  - *Orthogonality*: encourages isotropy in the learned projection  
- **Optimization**:  
  - Cosine schedule with warm-up  
  - Early stopping on `R@1` or `mAP`  

---

## Processing Pipeline

### 1. WebDataset Construction  
**Script:** `build_webdataset.py`

- Extracts multilingual image–caption pairs from **WIT** (Wikipedia Image Text, link to download -> https://github.com/google-research-datasets/wit/blob/main/DATA.md).  
- Filters entries covering **9 languages** (`ar, de, en, es, fr, it, ja, pt, zh`).  
- Validates and re-encodes images, saving shards as `.tar` in WebDataset format.

### 2. Image Validation  
**Script:** `scan_bad_images.py`  
Checks for corrupted or unreadable images inside shards.

### 3. Embedding Extraction
- **Images →** `emb_from_image.py`:  
  Generates CLIP-compatible image embeddings with OpenCLIP.
- **Texts →** `emb_from_text.py`:  
  Extracts multilingual CLIP-compatible text embeddings using SentenceTransformers.

### 4. Cross-Lingual Alignment  
**Script:** `align_text_to_image.py`

- Learns a lightweight projection ΔW that aligns all language embeddings to the shared image space.  
- Reports retrieval metrics (`R@{1,5,10}`, `mAP`) and representation diagnostics:
  - **Effective Rank**
  - **Isotropy (mean pairwise cosine)**
  - **Entropy & PoZ** (Percentage of Zero activations)
  - **Cross-lingual Gram Correlation**

### 5. Parameter Sweeps  
**Script:** `sweep_align.py`  
Runs multiple experimental configurations and aggregates performance tables.

---

## Evaluation

### Tasks
- **Text→Image / Image→Text retrieval**
- **Cross-lingual caption retrieval via image pivot**
- **Zero-shot classification** with multilingual prompts

### Metrics
- Recall@K (1, 5, 10)
- Mean Average Precision (mAP)
- Diagnostic measures:
  - Intrinsic Dimensionality
  - Gram Matrix Correlation
  - Entropy and isotropy statistics

---

## Representation Diagnostics
In line with reviewer suggestions, the project includes **deep structural analyses** of the learned embedding space:

- **Gram Correlation** — compares geometric similarity across languages  
- **Effective Rank / Intrinsic Dimensionality** — measures representational richness  
- **Entropy and PoZ** — higher entropy and lower PoZ indicate healthier, more expressive embeddings

---

## Repository Structure

```
├── build_webdataset.py        # Build multilingual WebDataset from WIT
├── scan_bad_images.py         # Detect and report invalid images
├── emb_from_image.py          # Compute image embeddings (OpenCLIP)
├── emb_from_text.py           # Compute multilingual text embeddings
├── align_text_to_image.py     # Learn cross-lingual alignment (ΔW)
├── sweep_align.py             # Parameter sweeps and result aggregation
├── playground.ipynb           # Tool to explore result and dataset
```

---

## ⚙️ Requirements

```bash
python >= 3.9
torch >= 2.0
sentence-transformers
open_clip_torch
webdataset
polars
pandas
scikit-learn
tqdm
Pillow
aiohttp
uvloop
```

---

## Quick Start

### 1. Build dataset
```bash
python build_webdataset.py --splits full
```

### 2. Generate embeddings
```bash
python emb_from_image.py --out_dir webdataset --splits full --device cuda
python emb_from_text.py  --out_dir webdataset --splits full --langs en,it,es,fr,de,pt,zh,ar,ja
```

### 3. Align multilingual spaces
```bash
python align_text_to_image.py   --out_dir webdataset --train_split full   --langs en,it,es,fr,de,pt,zh,ar,ja   --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 --save_report
```

### 4. Run full experimental sweep
```bash
python sweep_align.py
```

---

## Outputs
After training, the following artifacts are saved under `out_dir/alignment/`:
- `W_<split>_<langs>.pt` — learned projection matrix  
- `report_<split>_<langs>.json` — training and evaluation metrics  
- Diagnostic summaries with Gram correlations, entropy, isotropy, and rank statistics.

---

## Bibliography

1. Gella – *Image Pivoting for Learning Multilingual Multimodal Representations*  
2. Mohammadshahi – *Aligning Multilingual Word Embeddings for Cross-Modal Retrieval Task*  
3. Kim – *MULE: Multimodal Universal Language Embedding*  
4. Ni – *M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-Training*  
5. Zhou – *UC²: Universal Cross-Lingual Cross-Modal Vision-and-Language Pre-Training*  
6. Jain – *MURAL: Multimodal, Multitask Retrieval Across Languages*  
7. Chen – *mCLIP: Multilingual CLIP via Cross-lingual Transfer*  
8. Ahmat – *M²-VLP: Enhancing Multilingual Vision-Language Pre-Training via Multi-Grained Alignment*
