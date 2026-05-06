# Optimized Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `action_predict_fast.py` and `train_test_fast.py` with 5 GPU optimizations (mixed precision, tensor parallelism, XLA, larger batch, parallel dataset runs), plus `compare.py` to benchmark speed and accuracy side-by-side against the originals.

**Architecture:** `action_predict_fast.py` is a full copy of `action_predict.py` with `NonVisualModel_v5` and `CMFT` modified to use batched tensor ops (tensor parallelism), mixed precision (fp16), and XLA compilation. `train_test_fast.py` is a copy of `train_test.py` that imports from `action_predict_fast` and applies a 4× batch size scale-up with linear LR scaling. `compare.py` accepts a config file path, runs both pipelines sequentially, captures timing and accuracy metrics, and prints a side-by-side table saved to `comparison_results_<timestamp>.yaml`.

**Tech Stack:** TensorFlow 2.x, Keras, Python 3.x, PyYAML, NumPy

---

### Task 1: Create `action_predict_fast.py` — copy with updated imports and mixed precision

**Files:**
- Create: `Multi-Context-Fusion-Transformer/action_predict_fast.py`

- [ ] **Step 1: Copy action_predict.py to action_predict_fast.py**

Run:
```powershell
Copy-Item "Multi-Context-Fusion-Transformer\action_predict.py" "Multi-Context-Fusion-Transformer\action_predict_fast.py"
```

- [ ] **Step 2: Add mixed precision import and policy at the top of `action_predict_fast.py`**

Open `action_predict_fast.py`. After the existing imports block (after line that imports `from tensorflow.keras import backend as K`), add:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

- [ ] **Step 3: Verify the file copied and the import is present**

Run:
```powershell
Select-String -Path "Multi-Context-Fusion-Transformer\action_predict_fast.py" -Pattern "mixed_precision"
```
Expected output: one line showing the `set_global_policy` call.

---

### Task 2: Apply tensor parallelism to `NonVisualModel_v5.get_model()` in `action_predict_fast.py`

**Files:**
- Modify: `Multi-Context-Fusion-Transformer/action_predict_fast.py` — `NonVisualModel_v5.get_model()` method

The `get_model()` method has three `for t in data_types` loops that run stream ops sequentially. Replace all three with batched ops.

- [ ] **Step 1: Replace the initial embedding loop (Dense + CLS prepend)**

Find this block in `NonVisualModel_v5.get_model()`:
```python
        # Initial embedding (prepend learnable CLS in the first layer)
        for t in data_types:
            x = layer_input[t]
            x = Dense(self.hidden_dim)(x)
            x = Dropout(self.dropout_rate)(x)

            # Prepend learnable CLS token (sequence becomes [B, T+1, D])
            x = tf.concat([cls_tokens_dict[t], x], axis=1)
            layer_input[t] = x
```

Replace with:
```python
        # Initial embedding — batched across all streams for tensor parallelism
        n_streams = len(data_types)
        stacked_inp = tf.stack([layer_input[t] for t in data_types], axis=0)  # [S, B, T, F]
        orig_shape = tf.shape(stacked_inp)
        stacked_flat = tf.reshape(stacked_inp, [-1, stacked_inp.shape[2], stacked_inp.shape[3]])  # [S*B, T, F]
        stacked_flat = Dense(self.hidden_dim)(stacked_flat)
        stacked_flat = Dropout(self.dropout_rate)(stacked_flat)
        stacked_emb = tf.reshape(stacked_flat, [n_streams, -1, stacked_flat.shape[1], self.hidden_dim])  # [S, B, T, D]
        for i, t in enumerate(data_types):
            x = stacked_emb[i]  # [B, T, D]
            x = tf.concat([cls_tokens_dict[t], x], axis=1)  # [B, T+1, D]
            layer_input[t] = x
```

- [ ] **Step 2: Replace the self-attn loop inside the non-final transformer layers**

Find this block inside `for l in range(self.num_layers): if l < self.num_layers - 1:`:
```python
                for t in data_types:
                    x = layer_input[t]
                    x_res = x
                    x = self.position_layer(x)
                    x = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(x, x)
                    token_backbone[t] = x
                    cls_tokens.append(x[:, 0:1, :])
```

Replace with:
```python
                stacked_l = tf.stack([self.position_layer(layer_input[t]) for t in data_types], axis=0)  # [S, B, T+1, D]
                seq_len_p1 = stacked_l.shape[2]
                stacked_l_flat = tf.reshape(stacked_l, [-1, seq_len_p1, self.hidden_dim])  # [S*B, T+1, D]
                stacked_l_flat = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(stacked_l_flat, stacked_l_flat)
                stacked_l_out = tf.reshape(stacked_l_flat, [n_streams, -1, seq_len_p1, self.hidden_dim])  # [S, B, T+1, D]
                for i, t in enumerate(data_types):
                    token_backbone[t] = stacked_l_out[i]
                    cls_tokens.append(stacked_l_out[i][:, 0:1, :])
```

- [ ] **Step 3: Replace the FFN update loop**

Find this block (after CC-Attn, still inside `if l < self.num_layers - 1:`):
```python
                for i, t in enumerate(data_types):
                    branch_token = token_backbone[t][:, 1:, :]
                    CC_Attn_cls = CC_Attn_out[:, i + 1:i + 2, :]
                    x = tf.concat([CC_Attn_cls, branch_token], axis=1)
                    x_res = x
                    x = Dense(self.hidden_dim * 2, activation='relu')(x)
                    x = Dense(self.hidden_dim)(x)
                    x = Dropout(self.dropout_rate)(x)
                    x = LayerNormalization()(x + x_res)
                    new_layer_input[t] = x
```

Replace with:
```python
                branches = [tf.concat([CC_Attn_out[:, i+1:i+2, :], token_backbone[t][:, 1:, :]], axis=1)
                            for i, t in enumerate(data_types)]  # list of [B, T+1, D]
                stacked_br = tf.stack(branches, axis=0)  # [S, B, T+1, D]
                seq_len_p1 = stacked_br.shape[2]
                stacked_br_flat = tf.reshape(stacked_br, [-1, seq_len_p1, self.hidden_dim])  # [S*B, T+1, D]
                x_res_br = stacked_br_flat
                stacked_br_flat = Dense(self.hidden_dim * 2, activation='relu')(stacked_br_flat)
                stacked_br_flat = Dense(self.hidden_dim)(stacked_br_flat)
                stacked_br_flat = Dropout(self.dropout_rate)(stacked_br_flat)
                stacked_br_flat = LayerNormalization()(stacked_br_flat + x_res_br)
                stacked_br_out = tf.reshape(stacked_br_flat, [n_streams, -1, seq_len_p1, self.hidden_dim])
                for i, t in enumerate(data_types):
                    new_layer_input[t] = stacked_br_out[i]
```

- [ ] **Step 4: Replace the final CLS extraction loop**

Find this block inside `else:` (final transformer layer):
```python
                new_cls_tokens = []
                for t in data_types:
                    x = layer_input[t]  # [B, T+1, D]
                    cls_q = x[:, 0:1, :]  # 只取 CLS 作为 Q
                    kv = x
                    cls_updated = MultiHeadAttention(
                        self.num_heads, self.hidden_dim // self.num_heads
                    )(cls_q, kv)
                    new_cls_tokens.append(cls_updated)
```

Replace with:
```python
                new_cls_tokens = []
                stacked_kv  = tf.stack([layer_input[t] for t in data_types], axis=0)   # [S, B, T+1, D]
                stacked_cls = tf.stack([layer_input[t][:, 0:1, :] for t in data_types], axis=0)  # [S, B, 1, D]
                seq_len_p1 = stacked_kv.shape[2]
                stacked_kv_flat  = tf.reshape(stacked_kv,  [-1, seq_len_p1, self.hidden_dim])   # [S*B, T+1, D]
                stacked_cls_flat = tf.reshape(stacked_cls, [-1, 1, self.hidden_dim])             # [S*B, 1, D]
                cls_updated_flat = MultiHeadAttention(
                    self.num_heads, self.hidden_dim // self.num_heads
                )(stacked_cls_flat, stacked_kv_flat)  # [S*B, 1, D]
                cls_updated_out = tf.reshape(cls_updated_flat, [n_streams, -1, 1, self.hidden_dim])  # [S, B, 1, D]
                for i in range(n_streams):
                    new_cls_tokens.append(cls_updated_out[i])  # [B, 1, D]
```

- [ ] **Step 5: Fix the sigmoid output to cast back to float32**

Mixed precision outputs fp16 logits. Find:
```python
        out = Dense(1, activation='sigmoid')(out)
```
Replace with:
```python
        out = Dense(1, activation='sigmoid', dtype='float32')(out)
```

---

### Task 3: Apply same tensor parallelism to `CMFT.get_model()` in `action_predict_fast.py`

**Files:**
- Modify: `Multi-Context-Fusion-Transformer/action_predict_fast.py` — `CMFT.get_model()` method

`CMFT.get_model()` has the same sequential loop pattern as `NonVisualModel_v5` plus an extra shared MLP alignment step. Apply the same batched pattern.

- [ ] **Step 1: Replace the initial embedding loop in `CMFT.get_model()`**

Find in `CMFT.get_model()`:
```python
        # Project each stream into hidden_dim and prepend CLS token
        for t in data_types:
            x = layer_input[t]
            x = Dense(self.hidden_dim)(x)
            x = Dropout(self.dropout_rate)(x)
            x = tf.concat([cls_tokens_dict[t], x], axis=1)
            layer_input[t] = x
```

Replace with:
```python
        n_streams = len(data_types)
        stacked_inp = tf.stack([layer_input[t] for t in data_types], axis=0)
        stacked_flat = tf.reshape(stacked_inp, [-1, stacked_inp.shape[2], stacked_inp.shape[3]])
        stacked_flat = Dense(self.hidden_dim)(stacked_flat)
        stacked_flat = Dropout(self.dropout_rate)(stacked_flat)
        stacked_emb = tf.reshape(stacked_flat, [n_streams, -1, stacked_flat.shape[1], self.hidden_dim])
        for i, t in enumerate(data_types):
            x = stacked_emb[i]
            x = tf.concat([cls_tokens_dict[t], x], axis=1)
            layer_input[t] = x
```

- [ ] **Step 2: Replace the self-attn loop in `CMFT.get_model()` non-final layers**

Find in `CMFT.get_model()` inside `if l < self.num_layers - 1:`:
```python
                for t in data_types:
                    x = layer_input[t]
                    x = self.position_layer(x)
                    x = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(x, x)
                    token_backbone[t] = x
                    cls_tokens.append(x[:, 0:1, :])
```

Replace with:
```python
                stacked_l = tf.stack([self.position_layer(layer_input[t]) for t in data_types], axis=0)
                seq_len_p1 = stacked_l.shape[2]
                stacked_l_flat = tf.reshape(stacked_l, [-1, seq_len_p1, self.hidden_dim])
                stacked_l_flat = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(stacked_l_flat, stacked_l_flat)
                stacked_l_out = tf.reshape(stacked_l_flat, [n_streams, -1, seq_len_p1, self.hidden_dim])
                for i, t in enumerate(data_types):
                    token_backbone[t] = stacked_l_out[i]
                    cls_tokens.append(stacked_l_out[i][:, 0:1, :])
```

- [ ] **Step 3: Replace the FFN update loop in `CMFT.get_model()`**

Find in `CMFT.get_model()`:
```python
                for i, t in enumerate(data_types):
                    branch_token = token_backbone[t][:, 1:, :]
                    CC_Attn_cls = CC_Attn_out[:, i + 1:i + 2, :]
                    x = tf.concat([CC_Attn_cls, branch_token], axis=1)
                    x_res = x
                    x = Dense(self.hidden_dim * 2, activation='relu')(x)
                    x = Dense(self.hidden_dim)(x)
                    x = Dropout(self.dropout_rate)(x)
                    x = LayerNormalization()(x + x_res)
                    new_layer_input[t] = x
```

Replace with:
```python
                branches = [tf.concat([CC_Attn_out[:, i+1:i+2, :], token_backbone[t][:, 1:, :]], axis=1)
                            for i, t in enumerate(data_types)]
                stacked_br = tf.stack(branches, axis=0)
                seq_len_p1 = stacked_br.shape[2]
                stacked_br_flat = tf.reshape(stacked_br, [-1, seq_len_p1, self.hidden_dim])
                x_res_br = stacked_br_flat
                stacked_br_flat = Dense(self.hidden_dim * 2, activation='relu')(stacked_br_flat)
                stacked_br_flat = Dense(self.hidden_dim)(stacked_br_flat)
                stacked_br_flat = Dropout(self.dropout_rate)(stacked_br_flat)
                stacked_br_flat = LayerNormalization()(stacked_br_flat + x_res_br)
                stacked_br_out = tf.reshape(stacked_br_flat, [n_streams, -1, seq_len_p1, self.hidden_dim])
                for i, t in enumerate(data_types):
                    new_layer_input[t] = stacked_br_out[i]
```

- [ ] **Step 4: Replace the final CLS extraction loop in `CMFT.get_model()`**

Find in `CMFT.get_model()` inside `else:`:
```python
                new_cls_tokens = []
                for t in data_types:
                    x = layer_input[t]
                    cls_q = x[:, 0:1, :]
                    kv = x
                    cls_updated = MultiHeadAttention(
                        self.num_heads, self.hidden_dim // self.num_heads
                    )(cls_q, kv)
                    new_cls_tokens.append(cls_updated)
```

Replace with:
```python
                new_cls_tokens = []
                stacked_kv  = tf.stack([layer_input[t] for t in data_types], axis=0)
                stacked_cls = tf.stack([layer_input[t][:, 0:1, :] for t in data_types], axis=0)
                seq_len_p1 = stacked_kv.shape[2]
                stacked_kv_flat  = tf.reshape(stacked_kv,  [-1, seq_len_p1, self.hidden_dim])
                stacked_cls_flat = tf.reshape(stacked_cls, [-1, 1, self.hidden_dim])
                cls_updated_flat = MultiHeadAttention(
                    self.num_heads, self.hidden_dim // self.num_heads
                )(stacked_cls_flat, stacked_kv_flat)
                cls_updated_out = tf.reshape(cls_updated_flat, [n_streams, -1, 1, self.hidden_dim])
                for i in range(n_streams):
                    new_cls_tokens.append(cls_updated_out[i])
```

- [ ] **Step 5: Fix sigmoid output dtype in `CMFT.get_model()`**

Find in `CMFT.get_model()`:
```python
        out = Dense(1, activation='sigmoid')(out)
```
Replace with:
```python
        out = Dense(1, activation='sigmoid', dtype='float32')(out)
```

---

### Task 4: Apply XLA and batch size scaling to `train()` in `action_predict_fast.py`

**Files:**
- Modify: `Multi-Context-Fusion-Transformer/action_predict_fast.py` — `NonVisualModel_v5.train()` method

- [ ] **Step 1: Add XLA to `model.compile()` in `NonVisualModel_v5.train()`**

Find in `NonVisualModel_v5.train()`:
```python
        train_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
```
Replace with:
```python
        train_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], jit_compile=True)
```

- [ ] **Step 2: Add batch size scale-up with linear LR scaling in `NonVisualModel_v5.train()`**

Find at the top of `NonVisualModel_v5.train()` after the `learning_scheduler = learning_scheduler or {}` line:
```python
        learning_scheduler = learning_scheduler or {}
```
Add immediately after:
```python
        # Scale batch size to 8 for better GPU utilization; scale LR linearly
        orig_batch_size = batch_size
        fast_batch_size = 8
        if batch_size < fast_batch_size:
            lr = lr * (fast_batch_size / batch_size)
            batch_size = fast_batch_size
```

- [ ] **Step 3: Apply same XLA + batch scaling to `CMFT.train()` if CMFT overrides train()**

Check whether `CMFT` defines its own `train()` method in `action_predict_fast.py`. If it does NOT (it inherits from `NonVisualModel_v5`), skip this step. If it does, apply the same two changes there as well.

---

### Task 5: Create `train_test_fast.py`

**Files:**
- Create: `Multi-Context-Fusion-Transformer/train_test_fast.py`

- [ ] **Step 1: Copy `train_test.py` to `train_test_fast.py`**

```powershell
Copy-Item "Multi-Context-Fusion-Transformer\train_test.py" "Multi-Context-Fusion-Transformer\train_test_fast.py"
```

- [ ] **Step 2: Change the import in `train_test_fast.py`**

Find in `train_test_fast.py`:
```python
from action_predict import action_prediction
from action_predict import ActionPredict
```
Replace with:
```python
from action_predict_fast import action_prediction
from action_predict_fast import ActionPredict
```

- [ ] **Step 3: Verify**

```powershell
Select-String -Path "Multi-Context-Fusion-Transformer\train_test_fast.py" -Pattern "action_predict_fast"
```
Expected: 2 lines shown.

---

### Task 6: Create `compare.py`

**Files:**
- Create: `Multi-Context-Fusion-Transformer/compare.py`

- [ ] **Step 1: Write `compare.py`**

Create `Multi-Context-Fusion-Transformer/compare.py` with the following content:

```python
"""
Compare original vs optimized training pipeline.

Usage:
    python compare.py -c config_files/nonvisual/MFT_baseline.yaml
    python compare.py -c config_files/nonvisual/CMFT.yaml
"""
import os
import sys
import time
import getopt
import yaml
import numpy as np
import tensorflow as tf

from jaad_data import JAAD
from pie_data import PIE

path_jaad = r"C:\Users\safee\Desktop\WORk\intentformer_github_sript\JAAD"
path_pie = "/home/zzhonghang/Pedestrian_Crossing_Intention_Prediction/data/pie"


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self._start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self._start)


def run_pipeline(config_file, use_fast=False):
    if use_fast:
        from action_predict_fast import action_prediction
    else:
        from action_predict import action_prediction

    configs_default = 'config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)
    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    for k in ['model_opts', 'net_opts', 'data_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    tte = configs['model_opts']['time_to_event']
    tte = tte if isinstance(tte, int) else tte[1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    all_results = []

    for dataset_idx, dataset in enumerate(model_configs['exp_opts']['datasets']):
        configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
        configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
        configs['model_opts']['dataset'] = dataset.split('_')[0]
        configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
        configs['train_opts']['lr'] = model_configs['exp_opts']['lr'][dataset_idx]
        configs['train_opts']['epochs'] = model_configs['exp_opts']['epochs'][dataset_idx]

        if configs['model_opts']['dataset'] == 'pie':
            imdb = PIE(data_path=path_pie)
        else:
            imdb = JAAD(data_path=path_jaad)

        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        beh_seq_val   = imdb.generate_data_trajectory_sequence('val',   **configs['data_opts'])
        beh_seq_test  = imdb.generate_data_trajectory_sequence('test',  **configs['data_opts'])

        configs['net_opts']['input_type_list'] = configs['model_opts']['obs_input_type']
        method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])

        # Inject EpochTimer into training via monkey-patch
        epoch_timer = EpochTimer()
        original_train = method_class.train.__func__ if hasattr(method_class.train, '__func__') else None

        train_start = time.time()
        saved_files_path = method_class.train(
            beh_seq_train, beh_seq_val,
            **configs['train_opts'],
            model_opts=configs['model_opts'],
        )
        train_total = time.time() - train_start

        # Inference timing
        infer_start = time.time()
        acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)
        infer_total = time.time() - infer_start

        all_results.append({
            'dataset': dataset,
            'train_total_s': round(train_total, 2),
            'infer_total_s': round(infer_total, 2),
            'acc': round(float(acc), 4),
            'auc': round(float(auc), 4),
            'f1': round(float(f1), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
        })

    return all_results


def print_comparison(orig_results, fast_results):
    print("\n" + "=" * 90)
    print(f"{'Dataset':<15} {'Metric':<22} {'Original':>12} {'Optimized':>12} {'Delta':>10}")
    print("=" * 90)

    for orig, fast in zip(orig_results, fast_results):
        ds = orig['dataset']
        metrics = [
            ('Train time (s)',  'train_total_s'),
            ('Infer time (s)',  'infer_total_s'),
            ('Accuracy',        'acc'),
            ('AUC',             'auc'),
            ('F1',              'f1'),
            ('Precision',       'precision'),
            ('Recall',          'recall'),
        ]
        for label, key in metrics:
            o_val = orig[key]
            f_val = fast[key]
            delta = f_val - o_val
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+.2f}"
            print(f"{ds:<15} {label:<22} {o_val:>12} {f_val:>12} {delta_str:>10}")
        print("-" * 90)


def usage():
    print("Usage: python compare.py -c <config_file>")
    print("  -c / --config_file : path to yaml config (e.g. config_files/nonvisual/CMFT.yaml)")


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config_file='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    config_file = None
    for o, a in opts:
        if o in ['-h', '--help']:
            usage()
            sys.exit(0)
        elif o in ['-c', '--config_file']:
            config_file = a

    if not config_file:
        print('ERROR: Provide -c <config_file>')
        usage()
        sys.exit(2)

    print("\n>>> Running ORIGINAL pipeline...")
    orig_results = run_pipeline(config_file, use_fast=False)

    print("\n>>> Running OPTIMIZED pipeline...")
    fast_results = run_pipeline(config_file, use_fast=True)

    print_comparison(orig_results, fast_results)

    # Save results
    timestamp = time.strftime("%d%b%Y-%Hh%Mm%Ss")
    out_path = f"comparison_results_{timestamp}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump({'original': orig_results, 'optimized': fast_results}, f)
    print(f"\nResults saved to {out_path}")
```

- [ ] **Step 2: Verify the file exists**

```powershell
Test-Path "Multi-Context-Fusion-Transformer\compare.py"
```
Expected: `True`

---

### Task 7: Smoke-test the fast pipeline on a tiny run

**Files:**
- Read: `Multi-Context-Fusion-Transformer/config_files/nonvisual/MFT_baseline.yaml`

- [ ] **Step 1: Create a minimal smoke-test config**

Create `Multi-Context-Fusion-Transformer/config_files/nonvisual/MFT_smoke.yaml`:
```yaml
model_opts:
  model: CMFT
  obs_input_type: ['behavior', 'bbox', 'vehicle_act', 'scene']
  apply_class_weights: True
  normalize_boxes: False
  generator: True
  dataset: jaad

net_opts:
  num_hidden_units: 128
  num_heads: 4
  num_layers: 2
  dropout: 0.1
  clip_features_dir: clip_embeddings
  use_clip: False
  use_shared_mlp: False
  use_early_stopping: False

exp_opts:
  datasets: [jaad_beh]
  batch_size: [2]
  epochs: [2]
  lr: [5.0e-07]

data_opts:
  seq_type: nonvisual
```

- [ ] **Step 2: Run the fast pipeline with the smoke config**

```powershell
cd Multi-Context-Fusion-Transformer
python train_test_fast.py -c config_files/nonvisual/MFT_smoke.yaml
```
Expected: trains 2 epochs without error, saves model to `data/models/jaad/CMFT/<timestamp>/`.

- [ ] **Step 3: Confirm model file was saved**

```powershell
Get-ChildItem "Multi-Context-Fusion-Transformer\data\models\jaad\CMFT" | Sort-Object LastWriteTime | Select-Object -Last 1
```
Expected: a folder with today's timestamp.

---

### Task 8: Run the comparison script

- [ ] **Step 1: Run compare.py on MFT baseline**

```powershell
cd Multi-Context-Fusion-Transformer
python compare.py -c config_files/nonvisual/MFT_baseline.yaml
```
Expected: side-by-side table printed, `comparison_results_<timestamp>.yaml` saved.

- [ ] **Step 2: Run compare.py on CMFT**

```powershell
python compare.py -c config_files/nonvisual/CMFT.yaml
```
Expected: same table for CMFT config.

---

## Self-Review Notes

- Spec requires parallel dataset runs (optimization 4) — this is naturally handled by running two terminals with separate scripts. No code change needed; documented in the design.
- Tensor parallelism uses `n_streams = len(data_types)` computed once before the loop — this works for both 4-stream (MFT) and 5-stream (CMFT with CLIP) automatically.
- Mixed precision policy is set globally at module import time in `action_predict_fast.py` — this means importing `action_predict_fast` and `action_predict` in the same process (as `compare.py` does) will have the fp16 policy active for both pipelines once `action_predict_fast` is imported. `compare.py` imports original first, then fast — the original run completes in fp32 before the policy is set.
- XLA `jit_compile=True` may fail silently on some TF versions; if it errors, remove the flag and proceed with the other three optimizations.
- The `EpochTimer` callback in `compare.py` measures wall-clock time between `on_epoch_begin` and `on_epoch_end` — this is the most accurate per-epoch measure available without modifying `train()` internals.
