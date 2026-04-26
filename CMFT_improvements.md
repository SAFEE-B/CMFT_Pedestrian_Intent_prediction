# CMFT Improvement Plan
## Context: JAAD Crossing Intent Prediction, CMFT vs NonVisualModel_v5 baseline (75% acc)

---

## Identified Problems

1. **LR too low (5e-7)** — model barely trains; new components (CLIP projection, shared MLP) never learn meaningful weights
2. **Shared MLP is architecturally mismatched** — forces kinematic (binary/sparse) and visual (dense) streams through same weights, destroying modality-specific features
3. **CLIP temporal redundancy** — 16 near-identical consecutive frames give attention nothing to work with; no frame-to-frame variation in crop embeddings
4. **CLIP crop quality** — small distant pedestrian crops are weak CLIP inputs; scene context (road, traffic, crosswalk) is more informative for intent than person appearance
5. **No spatial grounding** — CLIP doesn't know where in the frame the pedestrian is; position relative to road/crosswalk is intent-relevant

---

## Fix 1: Increase Learning Rate
**Impact: High | Effort: 2 min**

Change in `config_files/nonvisual/CMFT.yaml`:
```yaml
lr: [1.0e-04, 1.0e-04]
```
Current `5e-7` is so low the model is essentially frozen. Do this first before any other change.

---

## Fix 2: Remove Shared MLP (Ablation)
**Impact: Medium | Effort: 5 min**

Run existing ablation config:
```
python train_test.py -c config_files/nonvisual/CMFT_clip_only.yaml
```
If this beats full CMFT, shared MLP is confirmed harmful — make `CMFT_clip_only` the main config.

---

## Fix 3: Mean-Pool CLIP Across Frames
**Impact: Medium | Effort: 30 min**

In `action_predict.py`, end of `_load_clip_sequence`, replace the return statement:
```python
seq_array = np.stack(seq, axis=0)          # [T, 512]
mean_emb = seq_array.mean(axis=0, keepdims=True)   # [1, 512]
return np.repeat(mean_emb, len(seq), axis=0)       # [T, 512]
```
Rationale: consecutive frames are nearly identical; mean-pooling collapses redundancy and gives the model a stable global appearance signal instead of confusing near-duplicate temporal vectors.

---

## Fix 4: Full-Frame CLIP Instead of Pedestrian Crop
**Impact: High | Effort: 2-3 hours (re-extraction)**

In `extract_clip_features.py`, change `__getitem__` to use the full camera frame:
```python
def __getitem__(self, idx):
    img_path, bbox, out_path = self.records[idx]
    img = cv2.imread(img_path)
    if img is None:
        return torch.zeros(3, 224, 224), out_path
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    tensor = self.preprocess(pil_img)
    return tensor, out_path
```
Then re-extract:
```
rm -r clip_embeddings/JAAD/
python extract_clip_features.py --dataset jaad
```
Rationale: full frame encodes road, traffic density, crosswalk presence, pedestrian position in scene — all directly relevant to crossing intent. Crop only shows the person's clothing/pose, which is weakly correlated with intent.

---

## Fix 5: Add Pedestrian Position Embedding to CLIP Vector
**Impact: Medium | Effort: 1 hour**

In `_load_clip_sequence`, after loading each `emb`, append normalized bbox center:
```python
cx = ((bbox[0] + bbox[2]) / 2) / frame_width   # normalize to [0,1]
cy = ((bbox[1] + bbox[3]) / 2) / frame_height
emb = np.concatenate([emb, [cx, cy]])           # [514]
```
Update `clip_embedding_dim: 514` in CMFT config and net_opts.

Rationale: CLIP is position-agnostic; a pedestrian near a crosswalk vs middle of road looks identical to CLIP. Appending normalized position adds spatial grounding cheaply.

---

## Execution Order

| Step | Action | Config/File | Est. Time |
|------|--------|-------------|-----------|
| 1 | Fix LR → run CMFT full | CMFT.yaml | 40 min |
| 2 | Run clip-only ablation | CMFT_clip_only.yaml | 40 min (parallel with 1) |
| 3 | Apply mean-pool → re-run | action_predict.py | 1 hour |
| 4 | Full-frame re-extraction → re-run | extract_clip_features.py | 2-3 hours |
| 5 | Position embedding if still stuck | action_predict.py + config | 1 hour |

Steps 1 and 2 can run in parallel on two terminals.

---

## Decision Criteria

- If Fix 1 alone beats 75%: LR was the bottleneck, proceed to ablations for further gains
- If Fix 2 (clip_only) beats Fix 1 (full CMFT): shared MLP is harmful, drop it permanently
- If Fix 4 (full-frame) beats Fix 3 (crop + mean-pool): scene context is what drives the gain
- If nothing beats 75%: CLIP visual features are not informative for this task/split; consider replacing with optical flow or scene segmentation features instead
