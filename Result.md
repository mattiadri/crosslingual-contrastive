# Report — Analysis of Multilingual Text–Image Alignment Experiments

## 1. Overview

This report summarizes a series of experiments aimed at aligning multilingual text embeddings with image embeddings.  
Several configurations of the alignment procedure were tested, varying in regularization strength, parameterization, and training dynamics.

The findings indicate that **using the image space as the alignment pivot is suboptimal**.  
While it can improve training metrics, it consistently reduces generalization on unseen data.  
The study also highlights the impact of dataset limitations, the challenges of cross-cultural representation, and the potential inadequacy of a purely linear alignment model.

---

## 2. Key Findings

### 2.1 Image Pivot Behavior

- The **image pivot** approach (aligning text embeddings directly to a fixed visual space) performs well **only when the projection matrix \( W \)** remains **very close to the identity**.  
  Once the mapping is forced to adapt more aggressively (more epochs or weaker regularization), the training metrics increase but the validation performance **collapses**, showing clear **overfitting**.

- With the conservative parameterization **\( W = I + \Delta W \)** and mild regularization, the optimal model remains nearly identical to the identity mapping.  
  The deviation \( \Delta W \) is small, and the holdout \( R@1 \) improves by only **~2 percentage points**.

These results suggest that the original multilingual text and image embeddings were already **largely coherent** in their respective spaces.  
Further “pulling” of one space toward the other harms generalization.

**In summary:** using the image space as a pivot adds little to no useful information and can introduce distortions that degrade the multilingual structure.

---

## 3. Conceptual Interpretation

### 3.1 Visual embeddings are not universally representative

Image embeddings (e.g., those from CLIP) are shaped by an **English-centric, domain-specific visual distribution**.  
Aligning texts from nine different languages to that space enforces a geometry primarily mediated by English visual semantics, reducing linguistic neutrality and distorting cross-lingual relationships.

### 3.2 A good pivot must be semantically neutral

An effective pivot should represent concepts shared across languages and modalities in a **neutral and domain-independent** way.  
The image space is not neutral: it constitutes a noisy, biased domain influenced by cultural and contextual priors present in the visual dataset.

### 3.3 The multilingual text space is already well-aligned

Diagnostic metrics confirm that the textual embeddings are inherently well structured:
- **Gram-correlation:** approximately 0.59 after alignment  
- **Effective rank:** high, indicating rich variance  
- **PoZ (near-zero activations):** low  
- **Average entropy:** high  

These results point to a stable and isotropic space across languages.  
Consequently, additional projection toward the visual space provides only marginal benefits.

---

## 4. Dataset and Cultural Limitations

The dataset used for alignment is **relatively small** and likely lacks sufficient **cultural and linguistic diversity**.  
Although nine languages were included (Arabic, German, English, Spanish, French, Italian, Japanese, Portuguese, Chinese), the image–text pairs predominantly reflect **Western and English-centric cultural contexts**.

This setting acts as a **cross-cultural stress test**:  
it exposes how models trained on biased visual corpora fail to generalize when languages encode concepts that do not map neatly onto Western imagery.  
The modest gains observed under these conditions further indicate that the dataset does not provide enough multimodal coverage to support deep semantic alignment.

Future work should therefore employ larger and more balanced datasets, with culturally diverse visual content and parallel textual descriptions across multiple linguistic and regional domains.

---

## 5. Model Limitations: The Linearity of \( W \)

The projection matrix \( W \) was modeled as a **linear transformation**.  
While this choice simplifies optimization and interpretation, it may be **too restrictive** to capture the complex, nonlinear relationships between text and image semantics.

Possible extensions include:
- **Nonlinear projection heads** (e.g., small MLPs) applied after the base embeddings.  
- **Kernelized or manifold-based mappings** to preserve geometry while allowing curved alignments.  
- **Adaptive or conditional projections**, where the transformation depends on language or content domain.

Exploring these nonlinear variants could reveal cross-modal correspondences that a purely linear operator cannot capture.

---

## 6. Empirical Conclusion

The experiments demonstrate that:

> **Using image embeddings as the alignment pivot is not advantageous** when multilingual text embeddings are already well calibrated.  
> Maintaining **intra-textual alignment** and using images only as an auxiliary signal (for regularization or diagnostics) is a more robust and generalizable strategy.

---

## 7. Proposed Extensions

### 7.1 Text-Pivot Alignment (English as a Reference Language)

A natural next step is to use the **English text embeddings** as the alignment pivot instead of the visual space.  
In this setting, the goal is to align other languages directly to the English embeddings:

$$ z'_{L_i} = W_i \, z_{L_i}, \quad \text{minimize } \| W_i z_{L_i} - z_{\text{en}} \|^2 $$

**Rationale:**
- English embeddings are already aligned with images in CLIP, serving as a semantically grounded anchor.  
- The pivot remains textual and thus linguistically neutral.  
- The overall multilingual geometry is preserved, improving consistency across languages.

**Expected outcome:** improved and more stable generalization, particularly for languages that are distant from the visual training domain (e.g., Arabic, Japanese, Chinese).

---

### 7.2 Symmetric Text–Image Alignment

Another promising direction is **symmetric alignment**, in which both modalities learn to project into a shared latent space.  
Two projections are trained simultaneously:

$$
\begin{cases}
z'_{\text{text}} = f_\theta(z_{\text{text}}) \\
z'_{\text{image}} = g_\phi(z_{\text{image}})
\end{cases}
$$

The objective minimizes a **bilateral contrastive loss**:

$$
L = \tfrac{1}{2} \big(L_{\text{text→img}} + L_{\text{img→text}}\big)
$$

This approach allows both encoders to adapt, meeting halfway in a joint embedding space rather than forcing one to mimic the other.

**Advantages:**
- The shared space is not biased toward the image domain.  
- Better cross-lingual and cross-modal generalization.  
- Improved geometric stability (isotropy and variance preservation).

This principle underlies models such as **mCLIP** (*Multilingual CLIP*, CVPR 2022) and **MURAL** (*Multimodal Universal Representations*, ACL 2022).

---

## 8. Comparative Summary

| Configuration | Δ R@1 (holdout) | Δ mAP (holdout) | Overfitting | Interpretation |
|----------------|----------------:|----------------:|-------------|----------------|
| **Image pivot, unconstrained W** | −13 pt | −11 pt | Strong | Forced alignment; overfitting on train, collapse on holdout |
| **Image pivot, W = I + ΔW (light regularization)** | **+2 pt** | **+1.8 pt** | None | Small corrective mapping; space already coherent |
| **Text pivot (English)** *(to be implemented)* | | | | Linguistically neutral pivot; better transferability |
| **Symmetric text↔image alignment** *(to be implemented)* | | | | Balanced shared space; improved modality coherence |

---

## 9. Overall Conclusion

- The **image-pivot** strategy leads to overfitting and degraded generalization; it is only useful when starting from completely misaligned spaces.  
- The **conservative parameterization** \( W = I + \Delta W \) with mild regularization is a safe and robust choice, preserving semantic relationships.  
- The **English text pivot** and **symmetric text–image alignment** represent the most promising directions for developing genuinely universal multimodal representations.  
- The current dataset, while multilingual, remains limited in size and cultural scope; larger and more balanced multimodal corpora are required to achieve full cross-cultural generalization.  
- Finally, the assumption of a **linear mapping \( W \)** may be overly simplistic — exploring **nonlinear or adaptive projections** is an important next step toward modeling the true complexity of multimodal semantics.

---

Author: Mattia Di Iorio

Project: Crosslingual Contrastive Alignment  

Date: November 2025