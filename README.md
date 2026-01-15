Multilingual CLIP Alignment via Image Pivoting
==============================================

Overview
--------
This project investigates multilingual multimodal alignment between text and images
using an image as a semantic pivot.
The goal is to map captions in different languages into a shared embedding space
without using parallel text during training.

The approach follows the CLIP paradigm (Contrastive Language–Image Pretraining),
where the visual encoder acts as a language-agnostic anchor, encouraging semantically
similar captions across languages to converge in the same space.
This setup tests whether contrastive supervision from image–text pairs alone can
induce meaningful cross-lingual alignment.


Goals and Hypothesis
-------------------
- Goal: fine-tune a CLIP-style model with multilingual image–caption pairs while
  avoiding any text-parallel supervision.
- Hypothesis: the image pivot is sufficient to align language-specific text encoders
  into a shared semantic space.
- Focus: representation analysis — how alignment evolves across languages before and
  after fine-tuning.


Model Architecture
------------------

Backbone
- Vision Encoder: OpenCLIP (e.g., ViT-B/32 pretrained on OpenAI or LAION)
- Text Encoder: Multilingual CLIP-compatible SentenceTransformer
  (clip-ViT-B-32-multilingual-v1)
- Projection Head: trainable linear mapping (ΔW) aligning multilingual text embeddings
  to the image space

In the main setting, both encoders are kept frozen and only the projection head is
trained, allowing controlled analysis of multilingual alignment effects.
A small residual MLP projection head is optionally evaluated for comparison, but is
treated as an alternative to the linear mapping rather than as a sequential stage.


Training Objective
------------------
- Contrastive loss: symmetric InfoNCE on image–text pairs with in-batch negatives
- Image pivoting: captions in different languages are sampled together via their
  shared image, but are never paired directly
- Regularization terms:
  - Prox-ID: penalizes deviation from identity (‖W − I‖²) for stability
  - Orthogonality: encourages isotropy in the learned projection
- Optimization:
  - Cosine schedule with warm-up
  - Early stopping on R@1 or mAP


Processing Pipeline
-------------------

1. WebDataset Construction
Script: build_webdataset.py

- Extracts multilingual image–caption pairs from WIT
  (Wikipedia Image Text:
   https://github.com/google-research-datasets/wit/blob/main/DATA.md)
- Filters entries covering 9 languages
  (ar, de, en, es, fr, it, ja, pt, zh)
- Validates and re-encodes images, saving shards as .tar files in WebDataset format


2. Image Validation
Script: scan_bad_images.py

- Checks for corrupted or unreadable images inside shards.


3. Embedding Extraction
- Images -> emb_from_image.py:
  Generates CLIP-compatible image embeddings with OpenCLIP.
- Texts -> emb_from_text.py:
  Extracts multilingual CLIP-compatible text embeddings using
  SentenceTransformers.


4. Cross-Lingual Alignment
Script: align_text_to_image.py

- Learns a lightweight projection ΔW that aligns all language embeddings to the
  shared image space.
- Reports retrieval metrics (R@1, R@5, R@10, mAP) and representation diagnostics:
  - Effective Rank
  - Isotropy (mean pairwise cosine similarity)
  - Entropy and PoZ (percentage of near-zero activations)
  - Cross-lingual Gram matrix correlation


5. Parameter Sweeps
Script: sweep_align.py

- Runs multiple experimental configurations and aggregates performance tables.


6. Representation Visualization
Script: do_umap.py

- Produces local UMAP visualizations around selected anchor captions.
- Visualizes the displacement of text embeddings from the pretrained space to the
  aligned spaces obtained with linear and MLP projections.
- Highlights how alignment emerges through small, local corrections rather than
  global rearrangements.


Evaluation
----------

Tasks
- Text-to-image and image-to-text retrieval
- Cross-lingual caption retrieval via image pivot

Metrics
- Recall@K (1, 5, 10)
- Mean Average Precision (mAP)
- Diagnostic measures:
  - Intrinsic Dimensionality
  - Gram Matrix Correlation
  - Entropy and isotropy statistics


Representation Diagnostics
--------------------------
In line with reviewer-style analysis, the project includes structural diagnostics of
the learned embedding space:

- Gram Correlation: compares geometric similarity across languages
- Effective Rank / Intrinsic Dimensionality: measures representational richness
- Entropy and PoZ: healthier embeddings exhibit higher entropy and lower sparsity


Repository Structure
--------------------
build_webdataset.py        Build multilingual WebDataset from WIT
scan_bad_images.py         Detect and report invalid images
emb_from_image.py          Compute image embeddings (OpenCLIP)
emb_from_text.py           Compute multilingual text embeddings
align_text_to_image.py     Learn cross-lingual alignment (ΔW / MLP)
do_umap.py                 Local UMAP visualization
sweep_align.py             Parameter sweeps and aggregation
playground.ipynb           Tool to explore results and dataset


Requirements
------------
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


Quick Start
-----------

1. Build dataset
python build_webdataset.py --splits full

2. Generate embeddings
python emb_from_image.py --out_dir webdataset --splits full --device cuda
python emb_from_text.py  --out_dir webdataset --splits full \
  --langs en,it,es,fr,de,pt,zh,ar,ja

3. Align multilingual spaces
python align_text_to_image.py \
  --out_dir webdataset --train_split full \
  --langs en,it,es,fr,de,pt,zh,ar,ja \
  --epochs 20 --steps_per_epoch 12 --batch_per_lang 32 \
  --save_report

4. Run full experimental sweep
python sweep_align.py


Outputs
-------
After training, the following artifacts are saved under out_dir/alignment/:
- W_<split>_<langs>.pt            Learned projection matrix
- report_<split>_<langs>.json     Training and evaluation metrics
- Diagnostic summaries including Gram correlations, entropy, isotropy,
  and intrinsic dimensionality statistics.


Bibliography
------------
1. Gella – Image Pivoting for Learning Multilingual Multimodal Representations
2. Mohammadshahi – Aligning Multilingual Word Embeddings for Cross-Modal Retrieval Task
3. Kim – MULE: Multimodal Universal Language Embedding
4. Ni – M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-Training
5. Zhou – UC²: Universal Cross-Lingual Cross-Modal Vision-and-Language Pre-Training
6. Jain – MURAL: Multimodal, Multitask Retrieval Across Languages
7. Chen – mCLIP: Multilingual CLIP via Cross-lingual Transfer
8. Ahmat – M²-VLP: Enhancing Multilingual Vision-Language Pre-Training via Multi-Grained Alignment