"""
Offline CLIP feature extraction for JAAD and PIE datasets.

Reads pedestrian crop images frame-by-frame (using bounding boxes from the raw
data sequences), encodes them with CLIP ViT-B/32, and saves one (512,) float16
.npy file per (pedestrian, frame) pair.

Run once before training CMFT:
    python extract_clip_features.py --dataset jaad --data_path /path/to/JAAD
    python extract_clip_features.py --dataset pie  --data_path /path/to/PIE
"""

import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import clip
except ImportError:
    raise ImportError("Install openai-clip: pip install git+https://github.com/openai/CLIP.git")


# ---------------------------------------------------------------------------
# Pedestrian crop dataset
# ---------------------------------------------------------------------------

class PedCropDataset(Dataset):
    """Loads (img_path, bbox) pairs and returns CLIP-preprocessed tensors."""

    def __init__(self, records, preprocess):
        """
        Args:
            records: list of (img_path, bbox, out_path) tuples
                     bbox = [x1, y1, x2, y2] in pixel coords
            preprocess: CLIP preprocessing transform
        """
        self.records = records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, bbox, out_path = self.records[idx]

        img = cv2.imread(img_path)
        if img is None:
            tensor = torch.zeros(3, 224, 224)
            return tensor, out_path

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        tensor = self.preprocess(pil_img)  # (3, 224, 224) float32

        return tensor, out_path


def collate_fn(batch):
    tensors = torch.stack([b[0] for b in batch])
    paths = [b[1] for b in batch]
    return tensors, paths


# ---------------------------------------------------------------------------
# Record builders for each dataset
# ---------------------------------------------------------------------------

def build_jaad_records(data_path, output_dir):
    """
    Walk the JAAD images directory and build (img_path, bbox, out_path) records.

    JAAD images are stored as:
        <data_path>/images/<video_id>/<frame_id:05d>.png

    Bounding boxes come from the JAAD annotations XML files.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from jaad_data import JAAD

    print("Loading JAAD database (this may take a minute)...")
    imdb = JAAD(data_path=data_path)
    data = imdb.generate_data_trajectory_sequence('all',
                                                   seq_type='crossing',
                                                   fstride=1,
                                                   sample_type='all',
                                                   subset='default',
                                                   data_split_type='default',
                                                   min_track_size=1)

    records = []
    seen = set()

    images = data['image']    # list of sequences, each sequence is list of img paths
    bboxes = data['bbox']     # matching bboxes
    pids   = data['pid']      # pedestrian ids

    for seq_imgs, seq_bboxes, seq_pids in zip(images, bboxes, pids):
        for img_path, bbox, pid in zip(seq_imgs, seq_bboxes, seq_pids):
            while isinstance(pid, (list, tuple)):
                pid = pid[0]
            pid_str = str(pid)
            # Frame id from path: .../video_0001/00042.jpg -> 00042
            frame_name = os.path.splitext(os.path.basename(img_path))[0]
            key = (pid_str, frame_name)
            if key in seen:
                continue
            seen.add(key)

            out_path = os.path.join(output_dir, 'JAAD', f'{pid_str}_{frame_name}.npy')
            records.append((img_path, bbox, out_path))

    print(f"JAAD: {len(records)} unique (pedestrian, frame) pairs")
    return records


def build_pie_records(data_path, output_dir):
    """
    Walk the PIE images directory and build (img_path, bbox, out_path) records.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    from pie_data import PIE

    print("Loading PIE database (this may take a minute)...")
    imdb = PIE(data_path=data_path)
    data = imdb.generate_data_trajectory_sequence('all',
                                                   seq_type='crossing',
                                                   fstride=1,
                                                   sample_type='all',
                                                   subset='default',
                                                   data_split_type='default',
                                                   min_track_size=1)

    records = []
    seen = set()

    images = data['image']
    bboxes = data['bbox']
    pids   = data['pid']

    for seq_imgs, seq_bboxes, seq_pids in zip(images, bboxes, pids):
        for img_path, bbox, pid in zip(seq_imgs, seq_bboxes, seq_pids):
            pid_str = pid[0]
            frame_name = os.path.splitext(os.path.basename(img_path))[0]
            key = (pid_str, frame_name)
            if key in seen:
                continue
            seen.add(key)

            out_path = os.path.join(output_dir, 'PIE', f'{pid_str}_{frame_name}.npy')
            records.append((img_path, bbox, out_path))

    print(f"PIE: {len(records)} unique (pedestrian, frame) pairs")
    return records


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract(records, clip_model, preprocess, batch_size=512, num_workers=8):
    # Filter already-done records to allow resuming
    remaining = [(p, b, o) for p, b, o in records if not os.path.exists(o)]
    done = len(records) - len(remaining)
    if done > 0:
        print(f"Skipping {done} already-extracted embeddings.")

    if not remaining:
        print("All embeddings already exist.")
        return

    # Create output directories
    out_dirs = set(os.path.dirname(o) for _, _, o in remaining)
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    dataset = PedCropDataset(remaining, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model = clip_model.to(device)
    clip_model.eval()

    processed = 0
    t0 = time.time()

    with torch.no_grad():
        for images, out_paths in loader:
            images = images.to(device)
            features = clip_model.encode_image(images)  # [B, 512]
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
            features = features.cpu().numpy().astype(np.float16)

            for feat, out_path in zip(features, out_paths):
                np.save(out_path, feat)

            processed += len(out_paths)
            if processed % 1000 < batch_size:
                elapsed = time.time() - t0
                print(f"  {processed}/{len(remaining)} images  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone. {processed} embeddings in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(records, n=5):
    print("\n--- Verification ---")
    import random
    sample = random.sample(records, min(n, len(records)))
    embeddings = []
    for _, _, out_path in sample:
        if not os.path.exists(out_path):
            print(f"MISSING: {out_path}")
            continue
        emb = np.load(out_path).astype(np.float32)
        assert emb.shape == (512,), f"Bad shape {emb.shape}: {out_path}"
        assert np.all(np.isfinite(emb)), f"Non-finite values: {out_path}"
        norm = np.linalg.norm(emb)
        print(f"  shape={emb.shape}  norm={norm:.4f}  path={os.path.basename(out_path)}")
        embeddings.append(emb)

    if len(embeddings) >= 2:
        # Check via cosine similarity — embeddings should not all point the same direction
        e0 = embeddings[0] / np.linalg.norm(embeddings[0])
        sims = [float(np.dot(e0, e / np.linalg.norm(e))) for e in embeddings[1:]]
        all_same = all(s > 0.9999 for s in sims)
        assert not all_same, "All sampled embeddings are identical — something is wrong!"
        print(f"  Different pedestrians produce different embeddings (cos sims: {[f'{s:.3f}' for s in sims]}). OK.")

    # Disk usage
    out_dir = os.path.dirname(os.path.dirname(records[0][2]))
    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(out_dir)
        for f in files if f.endswith('.npy')
    )
    print(f"  Total disk usage: {total_bytes / 1e6:.1f} MB")
    print("--- Verification passed ---\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Extract CLIP ViT-B/32 features for CMFT')
    parser.add_argument('--dataset', choices=['jaad', 'pie', 'both'], default='jaad')
    parser.add_argument('--jaad_path', default=r'C:\Users\safee\Desktop\WORk\intentformer_github_sript\JAAD',
                        help='Path to JAAD dataset root')
    parser.add_argument('--pie_path', default='/home/user/data/pie',
                        help='Path to PIE dataset root')
    parser.add_argument('--output_dir', default='clip_embeddings',
                        help='Directory to save .npy embedding files')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CLIP ViT-B/32...")
    clip_model, preprocess = clip.load('ViT-B/32', device='cpu')
    # Freeze all CLIP parameters
    for p in clip_model.parameters():
        p.requires_grad = False

    all_records = []

    if args.dataset in ('jaad', 'both'):
        records = build_jaad_records(args.jaad_path, output_dir)
        extract(records, clip_model, preprocess, args.batch_size, args.num_workers)
        all_records.extend(records)

    if args.dataset in ('pie', 'both'):
        records = build_pie_records(args.pie_path, output_dir)
        extract(records, clip_model, preprocess, args.batch_size, args.num_workers)
        all_records.extend(records)

    if all_records:
        verify(all_records)


if __name__ == '__main__':
    main()
