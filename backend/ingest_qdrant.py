"""Ingestion helpers: convert datasets to Qdrant points using CLIP and sentence-transformers.

Example usage:
    from qdrant_client import QdrantClient
    from backend.qdrant_utils import get_qdrant_client
    client = get_qdrant_client()
    ingest_datasets(client, collection_name='crop_health_knowledge', data_roots=['data/PlantVillage','data/IP102'])
"""
import os
from typing import List, Optional

try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
except Exception:
    raise ImportError('Please install transformers and pillow: pip install transformers pillow')

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from qdrant_client.http import models as rest
    from qdrant_client import QdrantClient
except Exception:
    raise ImportError('Please install qdrant-client: pip install qdrant-client')

import tqdm


class Embedders:
    def __init__(self, clip_model_name: str = 'openai/clip-vit-base-patch32', text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: Optional[str] = None):
        self.device = device
        # Load CLIP for image embeddings
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        if device:
            try:
                self.clip_model.to(device)
            except Exception:
                pass
        # Text embedder (384 dims recommended)
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers is required for semantic embeddings (pip install sentence-transformers)')
        self.text_model = SentenceTransformer(text_model_name)

    def image_to_visual(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors='pt')
        with self.clip_model.device if hasattr(self.clip_model, 'device') else nullcontext():
            outputs = self.clip_model.get_image_features(**inputs)
        vec = outputs.detach().cpu().numpy().squeeze().tolist()
        # normalize to unit length for cosine similarity
        return vec

    def text_to_semantic(self, text: str) -> List[float]:
        vec = self.text_model.encode(text, convert_to_numpy=True)
        return vec.tolist()


# Utility to discover image files under a root, returning (image_path,label,metadata)
def discover_images(root: str):
    """Discover images. Interpret immediate subfolders as labels when possible."""
    items = []
    if not os.path.exists(root):
        print(f"Data root not found: {root}")
        return items
    # If root has subdirs each as class label
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden
        rel = os.path.relpath(dirpath, root)
        if rel.startswith('.'):
            continue
        # treat files in dirpath
        label = os.path.basename(dirpath)
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append((os.path.join(dirpath, fn), label, {'source_root': root}))
    return items


def ingest_datasets(client: QdrantClient, collection_name: str = 'crop_health_knowledge', data_roots: Optional[List[str]] = None, embedders: Optional[Embedders] = None, batch_size: int = 64):
    """Read images from given roots, produce embeddings and upsert into Qdrant.

    Each point payload will contain:
      - 'label' : original dataset label
      - 'stress_type' : label (if available)
      - 'crop_name' : payload guess
      - 'severity' : optional int (default 1)
      - 'source' : dataset root
      - 'path' : relative image path
      - 'description' : short text description (expert label as sentence)

    The two named vectors 'visual' and 'semantic' are set.
    """
    if data_roots is None:
        data_roots = ['data/PlantVillage', 'data/IP102']
    embedders = embedders or Embedders()

    for root in data_roots:
        print(f"Discovering images under: {root}")
        items = discover_images(root)
        print(f"Found {len(items)} images in {root}")
        # Upsert in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            points = []
            for idx, (img_path, label, meta) in enumerate(batch):
                try:
                    visual_vec = embedders.image_to_visual(img_path)
                    desc = f"Image of {label}. Expert label: {label}."
                    sem_vec = embedders.text_to_semantic(desc)

                    payload = {
                        'label': label,
                        'stress_type': label,
                        'crop_name': 'unknown',
                        'severity': 1,
                        'source': root,
                        'path': os.path.relpath(img_path, root),
                        'description': desc,
                    }
                    # Construct point struct
                    point_id = f"{os.path.basename(root)}::{i+idx}::{os.path.basename(img_path)}"
                    points.append(rest.PointStruct(id=point_id, vector={'visual': visual_vec, 'semantic': sem_vec}, payload=payload))
                except Exception as e:
                    print('Failed embedding', img_path, e)
            if points:
                client.upsert(collection_name=collection_name, points=points)
                print(f"Upserted {len(points)} points into {collection_name}")
    print('Ingestion complete.')
