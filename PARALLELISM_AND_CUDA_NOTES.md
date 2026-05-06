# Parallelism and CUDA Concepts in This Codebase

This document covers every place where parallelism, distributed computing, or
CUDA-specific concepts are active — with exact file and line references.

---

## 1. Data-Level Parallelism — Batching

**Concept:** Process B samples simultaneously in one GPU kernel instead of one
at a time. This is the most fundamental form of parallelism in deep learning.

**How it works:** A Dense layer applied to a batch is a single matrix multiply:
`[B, T, D] × [D, D_out] → [B, T, D_out]`. The GPU assigns different rows of
the output to different CUDA cores and computes them all at the same time. B=8
costs almost the same wall time as B=1 because the GPU's thousands of cores are
all busy simultaneously. The only difference is one kernel launch dispatches 8x
the work.

### Where the batch size is set

**`action_predict_fast.py`, lines 1973–1978** — fast pipeline forces batch=8:
```python
fast_batch_size = 8
if batch_size < fast_batch_size:
    batch_size = fast_batch_size
```

**`action_predict.py`, lines 1978–1982** — original pipeline also forced to
batch=8 for fair comparison:
```python
target_batch_size = 8
if batch_size < target_batch_size:
    batch_size = target_batch_size
```

### Where the batched data is consumed

**`action_predict_fast.py`, line 2038–2044** — `model.fit()` receives the
DataGenerator which yields batches of 8. `batch_size=None` tells Keras that
the generator itself controls batch size:
```python
history = train_model.fit(
    x=data_train['data'][0],
    y=None if self._generator else data_train['data'][1],
    batch_size=None,
    epochs=epochs,
    ...
)
```

**`action_predict_fast.py`, line 2096** — inference at batch=1 (test set,
each sample predicted independently):
```python
y_pred = test_model.predict(X_test, batch_size=1, verbose=1).ravel()
```

### Where each batch is assembled

**`action_predict_fast.py`, lines 2463–2507** — `DataGenerator` class:
```python
class DataGenerator(Sequence):                         # line 2463
    def __len__(self):                                 # line 2489
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def __getitem__(self, index):                      # line 2497
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        X = self._generate_X(indices)
        y = self._generate_y(indices)
        return X, y
```

`Sequence` (imported at line 26) is `tf.keras.utils.Sequence`. Keras
recognises it and calls `__getitem__` in a background thread while the GPU
is computing the previous batch — CPU and GPU overlap.

---

## 2. Tensor-Level Parallelism — Multi-Head Attention

**Concept:** Multi-head attention splits the hidden dimension D into H
independent subspaces (heads). All H heads compute their attention scores and
weighted sums in parallel within a single batched matrix multiply.

**Config:** `num_heads=4`, `hidden_dim=128`, so each head has `key_dim=32`.
The Q/K/V projection is `[B, T, 128] × [128, 4*32]` — one matmul that covers
all 4 heads at once. The GPU computes all 4×B×T outputs simultaneously.

### NonVisualModel_v5 (MFT baseline, no CLIP)

**`action_predict_fast.py`, lines 1916–1917** — per-stream self-attention
(each stream has its own MHA layer with separate weights):
```python
x = MultiHeadAttention(self.num_heads, self.hidden_dim // self.num_heads)(x, x)
```
Called inside a `for t in data_types` loop. 4 separate MHA instances,
4 sequential kernel dispatches.

**`action_predict_fast.py`, lines 1922–1928** — cross-stream CC-Attention
(all stream CLS tokens attend to each other):
```python
CC_Attn_out = MultiHeadAttention(
    self.num_heads, self.hidden_dim // self.num_heads
)(CC_Attn_input, CC_Attn_input)
```

**`action_predict_fast.py`, lines 1944–1949** — final layer CLS extraction
(CLS token as query, full sequence as key/value):
```python
cls_updated = MultiHeadAttention(
    self.num_heads, self.hidden_dim // self.num_heads
)(cls_q, kv)
```

### CMFT variant (with CLIP, shared MLP)

Same pattern at **lines 2332, 2350–2355, 2372–2377** — identical MHA
instantiations for the 5-stream (behavior, bbox, vehicle_act, scene, CLIP)
version of the model.

### What the GPU does inside one MHA call

Given input `X` of shape `[B, T, D]`:
1. Project to Q, K, V: three matmuls `[B, T, D] × [D, H*kd]` → `[B, T, H*kd]`
2. Split heads: reshape to `[B, H, T, kd]`
3. Attention scores: `[B, H, T, kd] × [B, H, kd, T]` → `[B, H, T, T]`
4. Softmax across T dimension
5. Weighted sum: `[B, H, T, T] × [B, H, T, kd]` → `[B, H, T, kd]`
6. Output projection: reshape + `[B, T, H*kd] × [H*kd, D]`

Steps 3 and 5 are batched matmuls — B×H independent attention computations
all dispatched as one CUDA kernel. With B=8, H=4, that is 32 independent
attention computations running in parallel on the GPU.

---

## 3. Arithmetic Parallelism — Mixed Precision / Tensor Cores

**Concept:** NVIDIA Tensor Cores are specialized matrix-multiply units that
operate on float16 inputs and produce float32 accumulators. They run at
roughly 2× the throughput of regular CUDA cores for the same matmul. Mixed
precision routes all matmuls through Tensor Cores by keeping activations in
float16, while weights are stored in float32 to prevent gradient underflow.

### Where it is activated

**`action_predict_fast.py`, line 24–25** — sets the global Keras policy at
import time, before any model is built:
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

This one line affects every layer built after it in the same process. All
`Dense`, `MultiHeadAttention`, `LayerNormalization` layers will:
- Store their weights as float32
- Cast inputs to float16 before the forward pass
- Execute matmuls on Tensor Cores
- Cast outputs back to float32 for the optimizer weight update

**`action_predict_fast.py`, line 1960** — final sigmoid output forced to
float32 to prevent NaN in binary cross-entropy:
```python
out = Dense(1, activation='sigmoid', dtype='float32')(out)
```
Same at **line 2389** for the CMFT variant.

### Why the original pipeline does not use this

**`action_predict.py`** — no `mixed_precision` import or policy set anywhere.
All layers build in float32. Matmuls run on regular CUDA cores at half the
throughput.

### The NaN overflow problem

float16 max value is 65504. If gradients exceed this during backprop (which
happens with high LR because the chain rule multiplies many partial derivatives
together), they overflow to `inf`, then `nan`, which propagates forward through
all subsequent batches. This is why LR scaling was removed — at LR=1e-5
effective with batch=8, the gradients stay within float16 range. At 4× that
(4e-5), they do not.

---

## 4. Instruction-Level Parallelism — XLA JIT Compilation

**Concept:** XLA (Accelerated Linear Algebra) is a compiler that takes the
entire TensorFlow computation graph for one training step and generates a
single optimized GPU program. It fuses multiple operations into single kernels,
eliminating the dispatch gaps between them and improving memory locality.

### Where it is activated

**`action_predict_fast.py`, line 2002**:
```python
train_model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
    jit_compile=True
)
```

`jit_compile=True` is the only difference from the original's compile call at
**`action_predict.py`, line 2006**:
```python
train_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
```

### What XLA does

On the first batch, XLA traces the entire training step (forward pass → loss
→ backward pass → weight update) into a computation graph, then:

- **Operator fusion:** `Dense → Dropout → LayerNorm → residual add` becomes
  one kernel. Instead of 4 separate kernel launches (4 × ~10 µs dispatch
  overhead), one kernel runs with results staying in GPU registers between ops.
- **Memory layout optimization:** Tensor axis orderings are rewritten to match
  the GPU's preferred memory access patterns, reducing cache misses.
- **Constant folding:** Fixed tensors (e.g. position encodings that don't
  change between batches) are pre-computed and embedded in the compiled code.
- **Dead code elimination:** Any operations whose outputs are never used are
  removed before execution.

The first batch is slow (compilation takes a few seconds). Every subsequent
batch uses the compiled code. Over 40 epochs × 266–1076 steps per epoch, the
savings compound significantly.

### XLA system requirement

XLA needs CUDA's PTX math library to compile GPU code. It looks for it at
`<cuda_dir>/nvvm/libdevice/libdevice.10.bc`. On this machine it was only at
`Library\bin\`, requiring:
```powershell
New-Item -ItemType Directory -Force "...\Library\nvvm\libdevice"
Copy-Item "...\Library\bin\libdevice.10.bc" "...\Library\nvvm\libdevice\libdevice.10.bc"
$env:XLA_FLAGS = "--xla_gpu_cuda_data_dir=C:\Users\safee\anaconda3\envs\jaad_exp\Library"
```
`XLA_FLAGS` is forwarded to worker subprocesses via `env` in `_launch_one()`
at **`compare.py`, line 196–197**.

---

## 5. CPU-GPU Overlap — DataGenerator Prefetching

**Concept:** While the GPU is computing the forward+backward pass on batch N,
the CPU prepares batch N+1 in a background thread. The GPU never idles waiting
for data if the CPU is fast enough.

### Where this happens

**`action_predict_fast.py`, line 2463** — `DataGenerator(Sequence)`:

Keras's `model.fit()` internally spawns a background thread that calls
`DataGenerator.__getitem__()` ahead of time and places results in a queue.
The GPU training loop pulls from the queue. The default Keras settings when
a `Sequence` is passed are `workers=1, use_multiprocessing=False,
max_queue_size=10` — one thread (not a subprocess) pre-fetches up to 10
batches ahead. `use_multiprocessing=False` means the worker is a thread, not
a forked process; the prefetching itself is always active.

The key `Sequence` interface methods:
- `__len__` (line 2489): tells Keras how many batches exist per epoch
- `__getitem__` (line 2497): called by the background thread to fetch batch i
- `on_epoch_end` (line 2492): called after each epoch to re-shuffle indices

---

## 6. Process Isolation — Sequential Subprocesses

**Concept:** Not parallelism — sequential isolation. Each (pipeline, dataset)
combination runs in a completely separate Python process so that GPU memory,
CUDA context, and Python heap are fully released between runs.

### Where this happens

**`compare.py`, line 179** — `_launch_one()` function:
```python
def _launch_one(config_file, use_fast, dataset_idx, epochs_override, log_fh, xla_flags=''):
```

**`compare.py`, line 200–207** — subprocess launch and stream capture:
```python
proc = subprocess.Popen(
    cmd, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding='utf-8', errors='replace',
    bufsize=1,
)
for line in proc.stdout:          # streams output live to console + log
    sys.__stdout__.write(line)
    log_fh.write(line)
proc.wait()                       # blocks until subprocess exits
```

**`compare.py`, line 213** — exit code check:
```python
if proc.returncode != 0:
    raise RuntimeError(f"Worker exited with code {proc.returncode} ...")
```

**`compare.py`, lines 227–231** — outer loop spawning 4 processes total:
```python
def run_pipeline(config_file, use_fast, epochs_override, log_fh, xla_flags, n_datasets):
    results = []
    for idx in range(n_datasets):
        results.append(_launch_one(config_file, use_fast, idx, ...))
    return results
```

**`compare.py`, lines 333–340** — called twice (original then fast):
```python
orig_results = run_pipeline(config_file, use_fast=False, ...)
fast_results = run_pipeline(config_file, use_fast=True, ...)
```

### Why this was necessary

TensorFlow allocates a CUDA context when the first GPU operation runs and
holds it until the process exits. After training and testing jaad_all (6732
test samples, large CLIP arrays), the heap was fragmented and the CUDA context
held onto memory that the OS would not reclaim. The next run in the same
process then failed with either `MemoryError` or Windows access violation
`0xC0000005`. Subprocess exit = OS fully reclaims all GPU memory and RAM.

### CUDA concept — CUDA context

A CUDA context is the GPU-side equivalent of a process. It holds all allocated
device memory, compiled kernels, and stream state for one host process. Only
one context can be active per GPU thread at a time. When a subprocess exits,
its CUDA context is destroyed and all device memory is returned to the GPU's
free pool — equivalent to a full GPU reset for that client.

---

## 7. What Is NOT Used Here

| Concept | Why not applicable |
|---|---|
| **Multi-GPU / MirroredStrategy** | Single GPU — no replicas to synchronize |
| **NCCL / OpenMPI / Horovod** | Gradient averaging across multiple devices — irrelevant with one GPU |
| **Multiple CUDA streams** | Would allow concurrent kernel dispatch on one GPU, but TF/XLA uses a single stream by default. Requires low-level CUDA programming, not available through Keras |
| **Model parallelism** | Splitting layers across GPUs — not needed at this model size (~928K params) |
| **Python multiprocessing for speed** | Subprocesses here are sequential (for isolation), not parallel |
| **Async/await or threading for training** | Keras handles the prefetch thread internally — no manual threading code |
| **tf.distribute.Strategy** | Distribution across devices — only one device |

---

## Full Map

```
compare.py
├── run_pipeline(use_fast=False)          # original pipeline
│   ├── _launch_one(dataset_idx=0)        # subprocess 1: orig × jaad_beh
│   │   ├── CUDA context created
│   │   ├── model.fit() — 40 epochs
│   │   │   └── per step (266 steps/epoch):
│   │   │       ├── DataGenerator.__getitem__()    [CPU prefetch thread]
│   │   │       ├── Forward pass                   [GPU, float32]
│   │   │       │   ├── Dense layers               [batched matmul, B=8 parallel]
│   │   │       │   └── MultiHeadAttention ×N      [H=4 heads parallel per call]
│   │   │       ├── Loss + backward pass           [GPU]
│   │   │       └── Weight update                  [GPU]
│   │   ├── model.predict() — test set
│   │   └── subprocess exits → CUDA context destroyed, all memory freed
│   └── _launch_one(dataset_idx=1)        # subprocess 2: orig × jaad_all
│       └── [same as above, 1076 steps/epoch]
│
└── run_pipeline(use_fast=True)           # optimized pipeline
    ├── _launch_one(dataset_idx=0)        # subprocess 3: fast × jaad_beh
    │   ├── mixed_float16 policy set      [Tensor Cores active for all matmuls]
    │   ├── CUDA context created
    │   ├── model.fit() — 40 epochs
    │   │   └── per step (266 steps/epoch):
    │   │       ├── DataGenerator.__getitem__()    [CPU prefetch thread]
    │   │       ├── Forward pass                   [GPU, float16 matmuls on Tensor Cores]
    │   │       │   ├── Dense layers               [batched matmul, B=8 parallel, fp16]
    │   │       │   └── MultiHeadAttention ×N      [H=4 heads parallel, fp16]
    │   │       ├── XLA fused kernel               [no dispatch gaps between ops]
    │   │       ├── Loss + backward pass           [fp16 gradients]
    │   │       └── Weight update                  [fp32 master weights]
    │   ├── model.predict() — test set
    │   └── subprocess exits → memory freed
    └── _launch_one(dataset_idx=1)        # subprocess 4: fast × jaad_all
        └── [same as above, 1076 steps/epoch]
```
