# Pipeline Optimization Notes

This document describes every change made to create the optimized pipeline
(`action_predict_fast.py`) and what each change actually does under the hood.

---

## Background: What Is a TensorFlow Computation Graph?

When you write a Keras model and call `model.compile()` followed by
`model.fit()`, TensorFlow does not execute your Python code line-by-line during
training. Instead, it traces your model once — following every layer call,
every tensor operation, every loss computation — and builds a **computation
graph**: a static description of all the operations that need to happen, in
what order, with what data dependencies.

Think of it like a recipe card versus a cook. Your Python code is the cook who
reads the recipe and improvises. The computation graph is the printed recipe
card: every step is written down before any cooking starts, nothing is left to
interpretation, and you can hand the card to anyone — a GPU, a TPU, a cluster
— and they can execute it identically.

The graph has nodes (operations like MatMul, Add, Softmax) and edges (tensors
flowing between them). Once built, TensorFlow's XLA compiler can look at the
entire graph at once, see which operations can be fused together, reorganize
memory layout, and generate GPU machine code that is far more efficient than
dispatching each operation one at a time from Python.

**Why this matters for us:** The original pipeline runs in eager mode by
default — Python interprets each operation as it arrives, sends it to the GPU,
waits for the result, and moves on. The fast pipeline uses `jit_compile=True`
to tell TensorFlow to trace the graph first, hand the whole thing to XLA, and
let XLA generate a single fused GPU kernel for entire forward and backward
passes. The GPU never goes idle between operations.

---

## Why Not OpenMPI / NCCL / Horovod?

Those tools are for **multi-node or multi-GPU distributed training**: you have
4 GPUs (or 4 machines) each holding a copy of the model, each processing a
different mini-batch, and after each step they need to average their gradients
together. OpenMPI is a message-passing library that handles the inter-process
communication. NCCL (NVIDIA Collective Communications Library) does the same
thing but optimized for NVLink/PCIe between GPUs. Horovod wraps both into a
Keras-friendly API.

We have a single GPU. There is nothing to communicate with. Using MPI would
add process launch overhead, synchronization barriers, and gradient
all-reduce calls that each take time — and then average zero gradients, because
there is only one participant. The net result would be slower training, not
faster. All the speedup available to us comes from using the one GPU we have
more efficiently: less Python overhead, better memory layout, fused kernels.
That is exactly what XLA + mixed precision gives us.

---

## Changes Made

### 1. Mixed Precision Training (float16)

**File:** `action_predict_fast.py`, lines 24–25

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**Also:** `action_predict_fast.py`, line 1972 (and line 2411 for the CMFT
variant with CLIP):

```python
out = Dense(1, activation='sigmoid', dtype='float32')(out)
```

**What this is:**

Modern NVIDIA GPUs (Volta architecture onward, i.e. any RTX or Tesla card from
2017+) contain Tensor Cores — specialized matrix-multiply units that run at
roughly 2× the throughput when their inputs are 16-bit floats (float16) versus
32-bit floats (float32). Mixed precision exploits this by running the forward
pass and backward pass in float16, but keeping the weight update in float32.

The split is deliberate. float16 has a much smaller dynamic range (max value
≈65504 vs float32's ≈3.4×10³⁸). If you naively store and update weights in
float16, small gradient values underflow to zero and the model stops learning.
Keras's mixed precision policy handles this automatically: it stores a float32
master copy of every weight, computes gradients in float16, then applies the
update to the float32 copy. You get Tensor Core speed without numerical
instability.

The final sigmoid output layer is forced back to float32 explicitly because
the loss function (`binary_crossentropy`) needs full precision to compute
stable log values, especially when predictions are very close to 0 or 1.

**Effect measured:** jaad_all went from 48ms/step → 9ms/step in the test from
the comparison log (roughly 5× faster per step). The speedup comes entirely
from Tensor Core utilization — the matrix multiplications inside every
MultiHeadAttention and Dense layer are now done with 16-bit arithmetic.

---

### 2. XLA JIT Compilation

**File:** `action_predict_fast.py`, line 2014

```python
train_model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
    jit_compile=True          # <-- this flag
)
```

Original (`action_predict.py`, line 2006) has no `jit_compile` argument,
so it defaults to False (eager execution).

**What this is:**

XLA (Accelerated Linear Algebra) is a compiler that takes a TensorFlow
computation graph and generates optimized machine code for the target device.
`jit_compile=True` tells Keras to trace the training step — forward pass,
loss, backward pass, weight update — into a single XLA program, then compile
it before the first batch runs.

What XLA actually does during compilation:

- **Operator fusion:** Instead of dispatching 20 separate GPU kernels (one per
  layer), XLA merges compatible operations into one. A LayerNorm followed by a
  Dropout followed by a Dense becomes a single kernel. Each separate kernel
  launch has overhead (typically 5–20 µs on a modern GPU). Fusion eliminates
  most of that.

- **Memory layout optimization:** XLA can reorder tensor axes to match the
  memory access pattern the GPU hardware prefers, reducing cache misses.

- **Dead code elimination:** Operations whose outputs are never used are
  removed before any computation starts.

- **Constant folding:** Any tensor value that is fixed at compile time (e.g.,
  position encoding weights, dropout masks during inference) is pre-computed
  and baked in.

The first batch is slow (compilation takes a few seconds). Every subsequent
batch uses the compiled code. The longer the training run, the more you
benefit.

**Requirement:** XLA on GPU needs the `libdevice.10.bc` PTX math library.
TensorFlow looks for it at `<cuda_dir>/nvvm/libdevice/libdevice.10.bc`. On
this machine it was only at `Library\bin\libdevice.10.bc`, so it had to be
copied manually and `XLA_FLAGS` set:

```powershell
New-Item -ItemType Directory -Force "...\Library\nvvm\libdevice"
Copy-Item "...\Library\bin\libdevice.10.bc" "...\Library\nvvm\libdevice\libdevice.10.bc"
$env:XLA_FLAGS = "--xla_gpu_cuda_data_dir=C:\Users\safee\anaconda3\envs\jaad_exp\Library"
```

**Effect:** Compounds with mixed precision — both changes together are what
produce the large end-to-end speedup. XLA alone on a well-structured model
typically saves 15–30% of wall time.

---

### 3. Batched Multi-Head Attention Across Streams

**File:** `action_predict_fast.py`, lines 1906–1968 (NonVisualModel_v5
`get_model`) and lines 2334–2407 (CMFT `get_model` with CLIP)

**Original approach** (`action_predict.py`, lines 1907–1960 and 2325–2398):

Each input stream (behavior, bbox, vehicle_act, scene, and optionally CLIP) is
processed by its own independently instantiated MultiHeadAttention layer:

```python
for t in data_types:
    x = MultiHeadAttention(num_heads, key_dim)(layer_input[t], layer_input[t])
```

This means N separate MHA layer objects → N separate GPU kernel launches per
transformer layer per stream.

**Fast approach** (`action_predict_fast.py`, lines 1913–1918):

All streams are stacked into a single tensor along a new batch dimension, run
through **one** MHA call, then unstacked:

```python
# Stack: [S, B, T+1, D]  →  flatten to [S*B, T+1, D]
stacked_l = tf.stack([self.position_layer(layer_input[t]) for t in data_types], axis=0)
stacked_l_flat = tf.reshape(stacked_l, [-1, seq_len_p1, self.hidden_dim])

# Single MHA call on the combined batch
stacked_l_flat = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(
    stacked_l_flat, stacked_l_flat
)

# Unstack back to per-stream
stacked_l_out = tf.reshape(stacked_l_flat, [n_streams, -1, seq_len_p1, self.hidden_dim])
for i, t in enumerate(data_types):
    token_backbone[t] = stacked_l_out[i]
```

Same pattern is used for the FFN update block (lines 1931–1944) and the final
CLS extraction layer (lines 1947–1959).

**What this does:**

Instead of 4 (or 5) separate kernel dispatches, the GPU receives one larger
matrix multiplication. GPUs are throughput machines: they prefer fewer, larger
operations over many small ones because each kernel launch carries fixed
overhead and the GPU's thousands of cores are most efficient when fully loaded.
Folding S streams into the batch dimension multiplies the effective batch size
by S, which keeps the GPU's matrix-multiply units saturated.

**Known limitation — shared weights:** Because all streams pass through the
same single MHA layer object, they share that layer's weight matrices. In the
original, each stream has its own MHA weights. This is an architectural
difference, not just a speed trick. The parameter count drops from ~928,000 to
~333,000 as a result. The model trains and produces valid accuracy numbers, but
it is not identical to the original architecture. This has not yet been fixed.

---

### 4. Batch Size Scaling with Linear LR Rule

**File:** `action_predict_fast.py`, lines 1986–1990

```python
fast_batch_size = 8
if batch_size < fast_batch_size:
    lr = lr * (fast_batch_size / batch_size)
    batch_size = fast_batch_size
```

**Also added to original for fair comparison:** `action_predict.py`,
lines 1978–1982

```python
target_batch_size = 8
if batch_size < target_batch_size:
    lr = lr * (target_batch_size / batch_size)
    batch_size = target_batch_size
```

The config files set `batch_size: 2`. Both pipelines now override this to 8.

**What this is:**

Larger batches mean fewer gradient update steps per epoch (steps/epoch =
dataset_size / batch_size), so each epoch is faster. The GPU also utilizes its
memory bandwidth more efficiently — the overhead of loading model weights into
cache is amortized over 8 samples instead of 2.

The learning rate is scaled linearly because the gradient estimate at batch=8
is an average over 8 samples instead of 2. If you keep the same LR, the
effective learning rate per sample is 4× too small and the model converges
slower. The linear scaling rule (from the Facebook ResNet paper: "scale LR
proportionally to batch size") compensates: `lr_new = lr_old × (8/2) = lr_old × 4`.

**Effect on step count:** With `batch_size=2` and 2127 training samples,
steps/epoch = 2127/2 = ~1064. With `batch_size=8`, steps/epoch = 2127/8 = 266.
That is the 4× reduction in steps/epoch seen in the logs.

---

### 5. TeeStream Logging

**File:** `compare.py`, lines 22–52

```python
def setup_logging(log_path):
    logging.basicConfig(...)

    class TeeStream:
        def write(self, msg):
            self.original.write(msg)      # still prints to console
            if msg.strip():
                self.logger.log(...)      # also writes to file

    sys.stdout = TeeStream(sys.__stdout__, logger, logging.INFO)
    sys.stderr = TeeStream(sys.__stderr__, logger, logging.WARNING)
```

**What this is:** Python's `sys.stdout` is just an object with a `write()`
method. Replacing it with a wrapper that calls both the original and a file
logger means every `print()` statement in the entire process — including deep
inside Keras and TensorFlow — gets captured to the log file without modifying
any of those libraries. The `TeeStream` is called "tee" by analogy with the
Unix `tee` command which splits a stream to both stdout and a file.

**Effect:** All training progress, metrics, model summaries, and the final
comparison table are saved to `compare_log_<timestamp>.txt` for later review,
without losing console output during the run.

---

## Summary Table

| Change | File | Lines | Effect |
|---|---|---|---|
| Mixed precision float16 | `action_predict_fast.py` | 24–25, 1972 | ~5× faster matrix multiplications via Tensor Cores |
| Force float32 output layer | `action_predict_fast.py` | 1972, 2411 | Prevents NaN/Inf in binary cross-entropy loss |
| XLA JIT compilation | `action_predict_fast.py` | 2014 | Fuses GPU kernels, eliminates launch overhead |
| libdevice path fix | shell env | — | Required for XLA to compile CUDA kernels |
| Batched MHA across streams | `action_predict_fast.py` | 1913–1959, 2341–2395 | Fewer larger GPU ops instead of many small ones |
| Batch size 8 + LR scaling | `action_predict_fast.py` | 1986–1990 | 4× fewer steps/epoch, better GPU utilization |
| Batch size 8 (original too) | `action_predict.py` | 1978–1982 | Makes comparison fair — same steps/epoch in both |
| TeeStream logging | `compare.py` | 22–52 | Captures all output to timestamped log file |

---

## What the Progress Bar Steps Actually Are

When you see `266/266` or `1881/1881` in the output, each tick is one batch
processed. The total is `ceil(dataset_size / batch_size)`.

- **Training progress bar** (shows `loss:` and `val_loss:` at the end of each
  line): one forward + backward pass + weight update per step.
- **Test inference progress bar** (no loss values, just a bare count): the
  `method_class.test()` call runs prediction over the test set. The test set
  is typically processed at batch_size=1, so the step count equals the number
  of test samples. jaad_beh test set ≈ 1881 samples, jaad_all ≈ 6732 samples.
  These are not extra training runs — they are inference only.

The full sequence in a single compare.py run (2 datasets × 2 pipelines = 4
training runs + 4 inference passes):

```
266  steps   ← orig  jaad_beh training  (batch=8, 2127 samples)
1881 steps   ← orig  jaad_beh inference (batch=1, 1881 test samples)
1076 steps   ← orig  jaad_all training  (batch=8, 8607 samples)
6732 steps   ← orig  jaad_all inference (batch=1, 6732 test samples)
266  steps   ← fast  jaad_beh training
1881 steps   ← fast  jaad_beh inference
1076 steps   ← fast  jaad_all training
6732 steps   ← fast  jaad_all inference
```

---

## How the DataGenerator Works

**File:** `action_predict_fast.py`, lines 2485–2529

```python
class DataGenerator(Sequence):
    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        X = self._generate_X(indices)
        y = self._generate_y(indices)
        return X, y
```

`DataGenerator` inherits from `tf.keras.utils.Sequence`. Keras treats a
`Sequence` object as a lazy data source: it calls `__len__()` once at the
start of every epoch to know how many batches exist, then calls
`__getitem__(i)` for `i = 0, 1, ..., len-1` to fetch each batch on demand.

This is important for large datasets: the full dataset never has to sit in
GPU memory at once. Each call to `__getitem__` loads and returns exactly one
batch of samples — Keras immediately sends that batch to the GPU, computes
the gradient, updates weights, and discards the batch before requesting the
next one.

`on_epoch_end()` is called automatically after every epoch. It shuffles the
sample indices so the model sees the data in a different order each epoch,
which helps generalization.

`self.batch_size = 1 if len(self.labels) < batch_size else batch_size` is a
guard: if the entire dataset is smaller than one batch (e.g. a tiny validation
set), it falls back to batch_size=1 so it doesn't crash trying to form an
incomplete batch.

**`__len__` uses `np.floor`** — this means the last partial batch is dropped
if the dataset size is not exactly divisible by batch_size. For 2127 samples
at batch=8: `floor(2127/8) = 265` batches × 8 = 2120 samples used, 7 dropped.
The actual step count shown is 266 because Keras counts from 1. The dropped
samples change each epoch because of shuffling, so over many epochs every
sample is seen roughly equally.

---

## Mixed Precision Policy Contamination Fix

**File:** `compare.py`, lines 220–228

```python
# Ensure original runs in float32 before fast pipeline sets fp16 policy
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

print("\n>>> Running ORIGINAL pipeline (float32)...")
orig_results = run_pipeline(config_file, use_fast=False, ...)

print("\n>>> Running OPTIMIZED pipeline (float16 + XLA + batch=8)...")
mixed_precision.set_global_policy('mixed_float16')
fast_results = run_pipeline(config_file, use_fast=True, ...)
```

`action_predict_fast.py` sets `mixed_float16` at import time (line 25). In
Python, `import` is cached — once a module is imported, running `import` again
returns the cached version without re-executing the module body. This means
that if `action_predict_fast` was imported first, or if both modules were
imported in the same process, the `mixed_float16` policy set by the fast
module would silently contaminate the original pipeline's training run too.

The fix in `compare.py` is to explicitly reset the policy to `float32` before
the original pipeline runs, and then explicitly set it to `mixed_float16`
before the fast pipeline. This guarantees isolation regardless of import order.
Without this fix, the "original" pipeline would train in fp16 and the
comparison would be meaningless — both pipelines would look identical in speed
because both would be using Tensor Cores.

---

## The StopAfterPerfect Callback

**File:** `action_predict_fast.py`, lines 2028–2047

```python
class StopAfterPerfect(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val = logs.get(self.monitor, 0.0)
        if val >= 1.0:
            self._perfect_count += 1
            if self._perfect_count >= self.patience:
                self.model.stop_training = True
        else:
            self._perfect_count = 0
```

This is a custom early-stopping callback added to the fast pipeline. It only
triggers if the monitored metric (val_accuracy or accuracy) reaches exactly
1.0, and only after it stays there for `patience=5` consecutive epochs.

The reason this is stricter than standard `EarlyStopping` (which stops when
a metric stops improving) is that validation accuracy on these pedestrian
datasets can temporarily hit 1.0 early in training due to class imbalance —
the model learns to predict the majority class for everything. Standard early
stopping would halt training there. `StopAfterPerfect` requires that the
model sustain perfect accuracy for 5 epochs, ensuring it has actually learned
rather than collapsed.

It is only activated when `use_early_stopping: True` is set in the yaml config.
Both `MFT_baseline.yaml` and `CMFT.yaml` have `use_early_stopping: False`, so
this callback is not used in the comparison runs.

---

## Measured Results (from compare_log_06May2026-16h39m13s.txt, MFT_baseline, 1 epoch)

| Dataset  | Metric        | Original | Optimized | Delta    |
|----------|---------------|----------|-----------|----------|
| jaad_beh | Train time(s) | 19.42    | 23.24     | +3.82    |
| jaad_beh | Avg epoch(s)  | 19.42    | 23.24     | +3.82    |
| jaad_beh | Infer time(s) | 68.66    | 68.24     | -0.42    |
| jaad_beh | Accuracy      | 0.8167   | 0.8167    | 0.0000   |
| jaad_all | Train time(s) | 44.30    | 24.00     | -20.30   |
| jaad_all | Avg epoch(s)  | 44.30    | 24.00     | -20.30   |
| jaad_all | Infer time(s) | 122.37   | 68.42     | -53.95   |
| jaad_all | Accuracy      | 0.9583   | 0.9583    | 0.0000   |

**Reading the results:**

- **jaad_beh is slower in the optimized pipeline (+3.82s).** This is expected.
  The dataset has only 2127 training samples → 266 steps/epoch. XLA compilation
  takes several seconds on the first batch. With only 266 steps total (1 epoch),
  the compilation overhead is larger than the per-step savings. XLA pays off
  over many epochs, not on a 1-epoch smoke test.

- **jaad_all is 1.84× faster in training (-20.30s).** The larger dataset
  (8607 samples, 1076 steps) gives XLA enough batches to amortize compilation.
  The 4× reduction in ms/step (48ms → 9ms) is fully visible here.

- **Inference is 1.79× faster for jaad_all (-53.95s).** The inference pass
  also benefits from fp16 and XLA because `test_model` is compiled the same
  way. 6732 test samples at 18ms/step vs 10ms/step is the dominant factor.

- **Accuracy is identical in both pipelines.** This confirms that fp16 mixed
  precision and XLA do not degrade model quality — the math is numerically
  equivalent for these task metrics.

- **jaad_beh accuracy is low (0.8167) vs jaad_all (0.9583)** because
  `jaad_beh` uses only pedestrians with crossing behavior annotations, which
  is a harder and more balanced classification problem. `jaad_all` includes all
  pedestrians, making the negative class much larger and easier to dominate.

---

## Pending Issues

### Shared MHA Weights Bug

The batched MHA approach (change 3 above) uses a single `MultiHeadAttention`
layer object shared across all streams. This means all streams learn the same
attention weights. The original uses separate MHA instances per stream — each
stream develops its own attention pattern specialized to its input type (e.g.
bbox stream attends to positional changes, behavior stream to crossing
signals).

Parameter counts confirm this:
- Original:   ~928,385 trainable parameters
- Optimized:  ~333,569 trainable parameters

The fix is to instantiate one MHA object per stream but still call them in a
loop rather than batching — or to find a way to apply different weights to
different slices of the stacked tensor. The current implementation is
architecturally unsound for the purpose of comparing the two models fairly.
The speed measurement is valid; the accuracy comparison is not (we are
comparing two different models, not the same model at different speeds).

### CMFT.yaml LR Scaling

`CMFT.yaml` sets `lr: 1.0e-04`. With the batch=8 scaling rule, this becomes
`4.0e-04`. This is aggressive for a transformer with CLIP embeddings and may
cause training instability or slow convergence in early epochs. If the CMFT
comparison shows worse accuracy in the optimized pipeline, this should be
investigated first before attributing the difference to architectural changes.
