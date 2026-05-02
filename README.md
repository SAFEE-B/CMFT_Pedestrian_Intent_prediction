# CMFT: CLIP-Augmented Multi-Context Fusion Transformer

Pedestrian crossing intention prediction from non-visual (behavioral + contextual) features. Built on top of the Multi-Context Fusion Transformer (MFT), extended with a frozen CLIP semantic stream and a shared MLP alignment layer.

Evaluated on [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/).

---

## Architecture

![CMFT Architecture](cmft_arch.png)

CMFT fuses five context streams through a four-stage progressive attention pipeline:

| Stream | Features | Dim |
|--------|----------|-----|
| Pedestrian Behavior (P) | motion, gaze, head, gesture, direction | N×5 |
| Localization (L) | bounding box coordinates (top-left, bottom-right) | N×4 |
| Vehicle Motion (V) | motion state (categorical, JAAD) | N×1 |
| Environment (E) | lanes, crosswalk, traffic light, ... | N×8 |
| CLIP Semantic (C) *(new)* | ViT-B/32 full-frame + bbox centre, mean-pooled | N×514 |

**Four-stage pipeline:**

1. **ICF — Intra-Context Fusion**: Each stream passes through a Mutual-Attention (MI-Attn) Transformer encoder, producing a per-stream CLS token.
2. **Shared MLP Alignment** *(new)*: The same two-layer MLP (identical weights) is applied to every CLS token before cross-context fusion, aligning all streams into a shared representation space.
3. **CCF — Cross-Context Fusion**: A learnable global CLS token attends to all five context tokens via Multi-Context Attention (MC-Attn).
4. **ICR / CCR — Intra/Cross-Context Refinement**: Each stream is refined by Guided Intra-context Attention (GI-Attn), then a final Guided Cross-context Attention (GC-Attn) produces the global CLS.

The global CLS is passed to an MLP classifier with sigmoid output: P(crossing) ∈ [0, 1].

---

## Changes from MFT baseline

| Component | MFT | CMFT |
|-----------|-----|------|
| Input streams | 4 (P, L, V, E) | 5 (+ CLIP semantic C) |
| CLIP projection | — | Linear(514→128), frozen encoder |
| Shared MLP alignment | — | Dense(512, relu) → Dense(128), shared weights across all streams |
| Parameters | ~1.03M | ~1.29M |

---

## Results on JAAD

All runs: `hidden_dim=128, heads=4, layers=2, lr=1e-4, epochs=40`, overlap=0.8, threshold=0.5.

### jaad_beh (behavioral pedestrians only — harder split)

| Model | CLIP | Shared MLP | Acc | AUC | F1 | Precision | Recall |
|-------|------|------------|-----|-----|----|-----------|--------|
| MFT baseline | — | — | — | — | — | — | — |
| CLIP only | ✓ | — | 0.850 | 0.870 | 0.890 | 0.830 | 0.950 |
| Shared MLP only | — | ✓ | 0.800 | 0.740 | 0.850 | 0.820 | 0.880 |
| **CMFT (full)** | **✓** | **✓** | **0.860** | **0.810** | **0.900** | **0.830** | **0.970** |

### jaad_all (all pedestrians — easier split)

| Model | CLIP | Shared MLP | Acc | AUC | F1 | Precision | Recall |
|-------|------|------------|-----|-----|----|-----------|--------|
| MFT baseline | — | — | — | — | — | — | — |
| CLIP only | ✓ | — | 0.960 | 0.980 | 0.880 | 0.810 | 0.970 |
| Shared MLP only | — | ✓ | 0.960 | 0.970 | 0.880 | 0.830 | 0.940 |
| **CMFT (full)** | **✓** | **✓** | **0.940** | **0.970** | **0.850** | **0.780** | **0.920** |

---

## Setup

```bash
pip install -r requirements.txt
```

CLIP embeddings must be pre-extracted and placed in `clip_embeddings/` before training. The CLIP encoder (ViT-B/32) is used only for extraction — it is frozen and not trained.

---

## Training

**Full CMFT on JAAD:**
```bash
python train_test.py -c config_files/nonvisual/NonVisualModel.yaml
```

**Ablation studies:**
```bash
# MFT baseline (no CLIP, no shared MLP)
python train_test.py -c config_files/nonvisual/MFT_baseline.yaml

# CLIP stream only
python train_test.py -c config_files/nonvisual/CMFT_clip_only.yaml

# Shared MLP alignment only
python train_test.py -c config_files/nonvisual/CMFT_mlp_only.yaml
```

**On PIE dataset:**
```bash
python train_test.py -c config_files_pie/nonvisual/NonVisualModel.yaml
```

---

## Config options (`net_opts`)

| Key | Default | Description |
|-----|---------|-------------|
| `use_clip` | `True` | Include CLIP semantic stream |
| `use_shared_mlp` | `True` | Apply shared MLP alignment before CCF |
| `use_early_stopping` | `False` | Stop after acc=1.0 for 5 consecutive epochs (enabled for ablations) |
| `num_hidden_units` | `128` | Hidden dimension D |
| `num_heads` | `4` | Attention heads |
| `num_layers` | `2` | Transformer encoder layers per stream |
