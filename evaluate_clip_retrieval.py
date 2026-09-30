from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"


def load_gallery(features_path: Path):
    item_ids = []
    embeddings = []
    captions_per_item = []

    with h5py.File(features_path, "r") as h5:
        for key in sorted(h5.keys()):
            grp = h5[key]

            if "mean" not in grp:
                continue

            emb = np.asarray(grp["mean"], dtype=np.float32)

            if emb.ndim != 1:
                raise ValueError(f"{key}: expected mean embedding [D], got {emb.shape}")

            if not np.isfinite(emb).all():
                raise ValueError(f"{key}: embedding contains NaN/Inf")

            item_id = str(grp.attrs.get("item_id", key))

            raw_caps = grp.attrs.get("captions_json", "[]")
            if isinstance(raw_caps, bytes):
                raw_caps = raw_caps.decode("utf-8")

            try:
                captions = json.loads(raw_caps)
            except Exception:
                captions = []

            captions = [
                str(x).strip()
                for x in captions
                if str(x).strip()
            ]

            item_ids.append(item_id)
            embeddings.append(emb)
            captions_per_item.append(captions)

    if not embeddings:
        raise RuntimeError(f"No mean embeddings found in {features_path}")

    gallery = np.stack(embeddings, axis=0)

    # Normalize again for safety.
    norms = np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery = gallery / np.clip(norms, 1e-12, None)

    return item_ids, gallery, captions_per_item


class ClipTextEncoder:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size=64):
        out = []

        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding queries",
            unit="batch",
        ):
            batch = texts[start:start + batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    text_outputs = self.model.text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    feats = self.model.text_projection(text_outputs.pooler_output)
            else:
                text_outputs = self.model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                feats = self.model.text_projection(text_outputs.pooler_output)

            feats = F.normalize(feats.float(), p=2, dim=-1)
            out.append(feats.cpu())

        return torch.cat(out, dim=0).numpy()


def build_queries(item_ids, captions_per_item, max_queries=None):
    queries = []
    targets = []

    for item_index, captions in enumerate(captions_per_item):
        for caption in captions:
            queries.append(caption)
            targets.append(item_index)

            if max_queries is not None and len(queries) >= max_queries:
                return queries, np.asarray(targets, dtype=np.int64)

    return queries, np.asarray(targets, dtype=np.int64)


def compute_ranks(similarity, targets):
    """
    Rank is 1-based.
    """
    order = np.argsort(-similarity, axis=1)

    ranks = np.empty(len(targets), dtype=np.int64)

    for i, target in enumerate(targets):
        pos = np.where(order[i] == target)[0]
        if len(pos) != 1:
            raise RuntimeError(f"Target {target} not found exactly once")
        ranks[i] = int(pos[0]) + 1

    return ranks


def report_metrics(ranks, gallery_size):
    print()
    print("Retrieval metrics")
    print("-----------------")
    print(f"Gallery size: {gallery_size}")
    print(f"Queries:      {len(ranks)}")
    print(f"R@1:          {(ranks <= 1).mean() * 100:6.2f}%")
    print(f"R@5:          {(ranks <= 5).mean() * 100:6.2f}%")
    print(f"R@10:         {(ranks <= 10).mean() * 100:6.2f}%")
    print(f"MRR:          {(1.0 / ranks).mean():6.4f}")
    print(f"Median Rank:  {np.median(ranks):.1f}")

    if gallery_size <= 10:
        print()
        print(
            "[NOTE] With only 10 gallery items, R@10 will always be 100%. "
            "This run is only a smoke test."
        )


def show_examples(
    queries,
    targets,
    similarity,
    item_ids,
    topk=3,
    n_examples=5,
):
    print()
    print("Example rankings")
    print("----------------")

    topk = min(topk, len(item_ids))
    order = np.argsort(-similarity, axis=1)

    for qi in range(min(n_examples, len(queries))):
        print(f"\nQuery: {queries[qi]}")
        print(f"Target: {item_ids[targets[qi]]}")

        for rank, idx in enumerate(order[qi, :topk], start=1):
            marker = "*" if idx == targets[qi] else " "
            print(
                f"  {marker} {rank:>2}. "
                f"{item_ids[idx]} "
                f"(score={similarity[qi, idx]:.4f})"
            )


def validate_gallery(gallery):
    norms = np.linalg.norm(gallery, axis=1)

    print("Feature validation")
    print("------------------")
    print(f"Embedding shape:  {gallery.shape}")
    print(f"Norm min:         {norms.min():.6f}")
    print(f"Norm mean:        {norms.mean():.6f}")
    print(f"Norm max:         {norms.max():.6f}")
    print(f"Contains NaN:     {np.isnan(gallery).any()}")
    print(f"Contains Inf:     {np.isinf(gallery).any()}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--features",
        required=True,
        help="Path to clip_vit_b32_12f_mean.h5",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query limit for debugging.",
    )
    p.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of query examples to print.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k results to print for examples.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    features_path = Path(args.features)

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, using CPU.")
        device = "cpu"

    item_ids, gallery, captions_per_item = load_gallery(features_path)

    validate_gallery(gallery)

    queries, targets = build_queries(
        item_ids,
        captions_per_item,
        max_queries=args.max_queries,
    )

    if not queries:
        raise RuntimeError(
            "No captions found in HDF5. "
            "Check the captions_json attribute created during feature extraction."
        )

    print()
    print(f"Loaded {len(item_ids)} retrieval items")
    print(f"Loaded {len(queries)} caption queries")
    print(f"Text model: {MODEL_ID}")
    print(f"Device: {device}")

    encoder = ClipTextEncoder(device)
    query_embeddings = encoder.encode(
        queries,
        batch_size=args.batch_size,
    )

    # Both sides are normalized, so dot product == cosine similarity.
    similarity = query_embeddings @ gallery.T

    ranks = compute_ranks(similarity, targets)

    report_metrics(ranks, len(item_ids))

    show_examples(
        queries,
        targets,
        similarity,
        item_ids,
        topk=args.topk,
        n_examples=args.show,
    )


if __name__ == "__main__":
    main()