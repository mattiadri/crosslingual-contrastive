Multilingual CLIP Alignment via Image Pivoting
==============================================

Overview
--------
This project investigates multilingual multimodal alignment between text and images
using an image as a semantic pivot.

The objective is to map captions in different languages into a shared embedding space
without relying on parallel text or machine translation during training.

The approach follows the CLIP paradigm (Contrastive Language–Image Pretraining),
where the visual encoder acts as a language-agnostic anchor. By aligning multilingual
captions independently to the same image embedding, the model is encouraged to induce
cross-lingual semantic alignment purely through visual grounding.


Goals and Hypothesis
--------------------
- Goal: learn a shared multilingual text embedding space using only image–caption
  supervision, avoiding any text-parallel data.
- Hypothesis: the image pivot provides sufficient semantic signal to align
  language-specific text embeddings into a common space.
- Focus: representation analysis, emphasizing how alignment and geometry evolve
  before and after learning a lightweight projection.


Model Architecture
------------------

Backbone
- Vision Encoder: OpenCLIP (e.g., ViT-B/32 pretrained on OpenAI or LAION)
- Text Encoder: multilingual CLIP-compatible SentenceTransformer
  (sentence-transformers/clip-ViT-B-32-multilingual-v1)

Projection Head
- Linear projection ΔW (main setting)
- Optional residual MLP projection (alternative setting)

In the primary experimental setup, both the vision encoder and the text encoder are
kept frozen. Only the projection head is trained, enabling controlled analysis of
multilingual alignment effects without altering the pretrained representations.

Note: the MLP head is treated as an alternative to the linear mapping, not as a
sequential refinement stage.


Training Objective
------------------
- Contrastive loss: symmetric InfoNCE on image–text pairs with in-batch negatives.
- Image pivoting: captions in different languages are sampled via the same image,
  but are never paired directly with each other.
- Regularization:
  - Prox-ID: penalizes deviation from identity (‖W − I‖²) to preserve pretrained structure.
  - Orthogonality: optional constraint encouraging isotropy of the learned projection.
- Optimization:
  - Cosine learning rate schedule with warm-up.
  - Gradient clipping (optional).
  - Early stopping based on retrieval performance (e.g., R@1).


Processing Pipeline
-------------------

1. WebDataset Construction
Script: `build_webdataset.py`

- Extracts multilingual image–caption pairs from WIT (Wikipedia Image Text):
  https://github.com/google-research-datasets/wit/blob/main/DATA.md
- Filters entries covering 9 languages:
  ar, de, en, es, fr, it, ja, pt, zh
- Ensures no parallel text supervision is introduced.
- Validates and re-encodes images.
- Stores data as WebDataset shards (.tar).


2. Image Validation
Script: `scan_bad_images.py`

- Scans shards to detect corrupted or unreadable images.


3. Embedding Extraction
- Images → `emb_from_image.py`:
  Generates CLIP image embeddings using OpenCLIP.
- Texts → `emb_from_text.py`:
  Extracts multilingual text embeddings using a CLIP-compatible SentenceTransformer model.
- All embeddings are unit-normalized and cosine-ready.


4. Cross-Lingual Alignment
Script: `align_text_to_image.py`

- Learns a lightweight projection that aligns multilingual text embeddings to the shared
  image space.
- Supports:
  - Linear projection
  - Residual MLP projection
- Evaluation outputs:
  - Retrieval metrics (R@1, R@5, R@10, MRR)
  - Cross-lingual pivot retrieval (caption → image → caption in another language)
  - Extensive representation diagnostics (before vs after alignment)


5. Parameter Sweeps and Cross-Validation
Script: `sweep_align.py`

- Runs multiple configurations (linear vs MLP, regularization strength, temperature).
- Supports k-fold cross-validation.
- Aggregates results across folds into summary tables.


6. Representation Visualization
Script: `do_umap.py`

- Produces local UMAP visualizations around selected anchor captions.
- Compares:
  - Pretrained text embeddings
  - Linearly aligned embeddings
  - MLP-aligned embeddings
- Highlights that alignment typically emerges through small, local corrections rather than
  global rearrangements of the space.


Evaluation
----------

Tasks
- Text-to-image retrieval
- Image-to-text retrieval
- Cross-lingual caption retrieval via image pivot

Metrics
- Recall@K (1, 5, 10)
- Mean Reciprocal Rank (MRR)

Evaluation is performed both per-language and as macro-averages across languages.


Representation Diagnostics
--------------------------
The project places strong emphasis on structural analysis of the embedding space:

- Gram Matrix Correlation:
  measures geometric similarity across languages.
- Neighbor Overlap:
  evaluates cross-lingual neighborhood consistency.
- Effective Rank / Intrinsic Dimensionality:
  quantifies representational richness.
- Isotropy:
  mean pairwise cosine similarity.
- Entropy and PoZ (percentage of near-zero activations):
  indicators of sparsity and collapse.
- Language-ID Probing:
  linear probe accuracy to assess residual language-specific information.

All diagnostics are reported before and after alignment, enabling fine-grained analysis
of representational changes.


Repository Structure
--------------------
- `build_webdataset.py`: Build multilingual WebDataset shards from WIT
- `scan_bad_images.py`: Detect and report corrupted or unreadable images
- `emb_from_image.py`: Compute image embeddings using OpenCLIP
- `emb_from_text.py`: Compute multilingual text embeddings
- `align_text_to_image.py`: Learn cross-lingual alignment
- `sweep_align.py`: Parameter sweeps and cross-validation
- `do_umap.py`: UMAP visualization
- `do_tables.py`: Aggregate metrics and generate tables
- `playground.ipynb`: Interactive exploration
- `requirements.txt`: Python dependencies


Requirements
------------
```

pandas
polars
torch
requests
tqdm
pyarrow
aiohttp
pillow
webdataset
uvloop
orjson
torchvision
timm
cairosvg
open_clip_torch
sentence_transformers
matplotlib
tabulate
umap-learn

```


Quick Start
-----------

1. Build the dataset
```

python build_webdataset.py --splits full

```

2. Generate embeddings
```

python emb_from_image.py --out_dir webdataset --splits full --device cuda

python emb_from_text.py --out_dir webdataset --splits full 
--langs en,it,es,fr,de,pt,zh,ar,ja

```

3. Align multilingual text to image space
```

python align_text_to_image.py 
--out_dir webdataset 
--train_split full 
--langs en,it,es,fr,de,pt,zh,ar,ja 
--epochs 20 
--steps_per_epoch 12 
--batch_per_lang 32 
--save_report

```

4. Run full sweep with cross-validation
```

python sweep_align.py

```


Outputs
-------
All outputs are stored under `out_dir/alignment/`:

- `W_<split>_<langs>[_foldK]_<head>.pt`
  Learned projection weights.
- `report_<split>_<langs>[_foldK]_<head>.json`
  Detailed metrics and diagnostics.
- Aggregated tables and summaries from cross-validation runs.


Bibliography
------------
1. Gella – Image Pivoting for Learning Multilingual Multimodal Representations
2. Mohammadshahi – Aligning Multilingual Word Embeddings for Cross-Modal Retrieval
3. Kim – MULE: Multimodal Universal Language Embedding
4. Ni – M3P: Multilingual Multimodal Pre-Training
5. Zhou – UC²: Universal Cross-Lingual Cross-Modal Vision-and-Language Pre-Training
6. Jain – MURAL: Multimodal, Multitask Retrieval Across Languages
7. Chen – mCLIP: Multilingual CLIP via Cross-lingual Transfer
8. Ahmat – M²-VLP: Multi-Grained Multilingual Vision-Language Alignment