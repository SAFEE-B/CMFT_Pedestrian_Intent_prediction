import os
import numpy as np

d = 'clip_embeddings/JAAD'
files = [f for f in os.listdir(d) if f.endswith('.npy')]
total = len(files)
print(f"Normalizing {total} embeddings...")

for i, f in enumerate(files):
    p = os.path.join(d, f)
    e = np.load(p).astype(np.float32)
    e = e / np.linalg.norm(e)
    np.save(p, e.astype(np.float16))
    if i % 10000 == 0:
        print(f"  {i}/{total}")

print("Done")
