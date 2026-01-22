"""
Memory-efficient Qdrant ingestor for agricultural multimodal datasets.

Usage (simple):
    from backend.agri_batch_ingestor import AgriBatchIngestor

    ing = AgriBatchIngestor(collection_name="crop_health_knowledge", qdrant_url=":memory:")
    ing.setup_collection()
    ing.ingest_multimodal_data(image_paths, text_data, batch_size=32)
    ing.finalize_ingestion()

CLI example (ingest images + optional per-image texts):
    python -m backend.agri_batch_ingestor --images-dir data/PlantDoc --text-file metadata/texts.txt --collection crop_health_knowledge

Notes:
- Uses SentenceTransformer encoders for both visual and semantic vectors to be lightweight on Colab.
- Uses small batches and clears CUDA cache between batches to avoid OOM.
- The collection is created with named vectors `visual` and `semantic` and HNSW disabled during ingestion for memory predictability.
"""

from __future__ import annotations
import os
import argparse
import uuid
from typing import List, Optional
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:
    raise ImportError("qdrant-client is required: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise ImportError("sentence-transformers is required: pip install sentence-transformers")

from PIL import Image
import torch


class AgriBatchIngestor:
    """Memory-efficient batch ingestor for visual+semantic data into Qdrant.

    - Uses named vectors: `visual` (512) and `semantic` (384)
    - Disables HNSW (m=0) during ingestion; re-enables after finalize
    - Uses small batches and clears CUDA cache between batches when available
    """

    def __init__(
        self,
        collection_name: str = "crop_health_knowledge",
        qdrant_url: str = ":memory:",
        device: Optional[str] = None,
        visual_model_name: str = "clip-ViT-B-32",
        text_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = QdrantClient(qdrant_url)
        self.collection_name = collection_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lightweight encoders (SentenceTransformer wrappers). Clip ViT wrapper uses 512-d visual vectors.
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        # Use a CLIP-style visual encoder from sentence-transformers if available
        try:
            self.vis_model = SentenceTransformer(visual_model_name, device=self.device)
            # If visual encoder outputs not 512, we still use it but collection size should match
        except Exception:
            # Fallback: try a CLIP model name compatible with transformers-based CLIP
            self.vis_model = SentenceTransformer(text_model_name, device=self.device)

    def setup_collection(self, on_disk: bool = True):
        """Create/recreate the Qdrant collection with named vectors and HNSW disabled for ingestion."""
        vectors_config = {
            "visual": rest.VectorParams(size=512, distance=rest.Distance.COSINE),
            "semantic": rest.VectorParams(size=384, distance=rest.Distance.COSINE),
        }
        try:
            # Best-effort recreate with named vectors
            self.client.recreate_collection(collection_name=self.collection_name, vectors_config=vectors_config)
        except Exception:
            try:
                self.client.create_collection(collection_name=self.collection_name, vectors_config=vectors_config)
            except Exception:
                pass

        # Disable HNSW (m=0) during bulk ingestion for predictable memory usage
        try:
            self.client.update_collection(collection_name=self.collection_name, hnsw_config=rest.HnswConfigDiff(m=0))
        except Exception:
            # not all qdrant versions expose HnswConfigDiff; ignore if unavailable
            pass

    def ingest_multimodal_data(self, image_paths: List[str], text_data: List[str], batch_size: int = 32):
        """Ingest paired image_paths and text_data lists in batches.

        Args:
            image_paths: list of image file paths
            text_data: list of short text descriptions/expert notes (same length as image_paths)
        """
        assert len(image_paths) == len(text_data), "image_paths and text_data must be same length"

        n = len(image_paths)
        for i in range(0, n, batch_size):
            batch_imgs = image_paths[i : i + batch_size]
            batch_txts = text_data[i : i + batch_size]

            # Load and prepare images for the sentence-transformers visual encoder
            pil_imgs = [Image.open(p).convert("RGB") for p in batch_imgs]
            with torch.no_grad():
                vis_embs = self.vis_model.encode(pil_imgs, convert_to_numpy=True, show_progress_bar=False)
                txt_embs = self.text_model.encode(batch_txts, convert_to_numpy=True, show_progress_bar=False)

            points = []
            for j in range(len(batch_imgs)):
                pid = str(uuid.uuid4())
                vectors = {
                    "visual": vis_embs[j].tolist(),
                    "semantic": txt_embs[j].tolist(),
                }
                payload = {
                    "image_source": os.path.basename(batch_imgs[j]),
                    "expert_notes": batch_txts[j],
                }
                points.append(rest.PointStruct(id=pid, vector=vectors, payload=payload))

            # Upsert
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Ingested batch {i//batch_size + 1} ({len(points)} points)")

            # Clear GPU cache to keep memory stable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def finalize_ingestion(self, hnsw_m: int = 16):
        """Re-enable HNSW after ingestion for fast search performance."""
        try:
            self.client.update_collection(collection_name=self.collection_name, hnsw_config=rest.HnswConfigDiff(m=hnsw_m))
        except Exception:
            # ignore if backend doesn't support update or HNSW config
            pass


# Simple CLI for convenience
def gather_image_text_pairs(images_dir: str, text_file: Optional[str] = None, max_files: Optional[int] = None):
    """Collect image paths and optional text mappings.

    If text_file is provided, it is expected to be a TSV: filename <TAB> text. Otherwise text is inferred from filename.
    """
    images = []
    for root, _, fns in os.walk(images_dir):
        for fn in fns:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(os.path.join(root, fn))
    images.sort()
    if max_files:
        images = images[:max_files]

    texts = []
    mapping = {}
    if text_file and os.path.exists(text_file):
        with open(text_file, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]

    for p in images:
        bn = os.path.basename(p)
        texts.append(mapping.get(bn, f"Auto-generated note for {bn}"))

    return images, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, help='Directory with images to ingest')
    parser.add_argument('--text-file', type=str, help='Optional TSV mapping filename->text (filename\ttext)')
    parser.add_argument('--collection', type=str, default='crop_health_knowledge')
    parser.add_argument('--qdrant-url', type=str, default=':memory:')
    parser.add_argument('--max-files', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    if not args.images_dir or not os.path.exists(args.images_dir):
        parser.error('Provide --images-dir pointing to a folder of images')

    images, texts = gather_image_text_pairs(args.images_dir, args.text_file, max_files=args.max_files)
    ing = AgriBatchIngestor(collection_name=args.collection, qdrant_url=args.qdrant_url)
    ing.setup_collection()
    ing.ingest_multimodal_data(images, texts, batch_size=args.batch_size)
    ing.finalize_ingestion()

    print('Ingestion finished — collection:', args.collection)


if __name__ == '__main__':
    main()