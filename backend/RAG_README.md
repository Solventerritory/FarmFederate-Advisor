# RAG Agent (Qdrant + Raptor Mini) — Notes

Quick steps to get started:

1. Install added dependencies:

    pip install -r backend/requirements_rag.txt

2. Start Qdrant locally (docker):

    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

3. (Optional) Set environment variables:

    - QDRANT_URL (default: http://localhost:6333)
    - QDRANT_API_KEY
    - HF_TOKEN (Hugging Face token) to call Raptor Mini inference

4. Initialize collections and run an ingestion demo:

    python backend/examples/rag_demo.py --init --ingest

5. Run a diagnosis on a sample image:

    python backend/examples/rag_demo.py --image data/PlantVillage/some_folder/image.jpg --desc "yellow spots on leaves" --farm FARM1 --plant PLANT1


Notes
- The CLIP visual embeddings are generated via `transformers` CLIP model (default `openai/clip-vit-base-patch32`).
- Semantic embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- `call_raptor_mini()` uses the Hugging Face Inference API if `HF_TOKEN` is set; otherwise it returns an offline placeholder. Replace with any LLM of choice if needed.

Colab-friendly usage

- For Colab (T4 / 16GB), prefer the lightweight module `backend/qdrant_rag_colab.py` which:
  - Uses `QdrantClient(':memory:')` when `USE_QDRANT_LOCAL=1` (fast in-memory testing), or a local path via `QDRANT_PATH`.
  - Defaults to small `RAG_BATCH_SIZE` and optional `RAG_FP16=1` when CUDA is available to reduce memory use.
  - Provides `init_colab_collections()`, `small_ingest_sample()`, `agentic_diagnose_colab()` and `store_session_colab()` for quick demos.

Quick Colab test (copy & paste into a Colab cell):

```python
!pip install -q qdrant-client transformers sentence-transformers torch pillow
from qdrant_client import QdrantClient
from backend.qdrant_rag_colab import init_colab_collections, agentic_diagnose_colab, small_ingest_sample, ColabEmbedders
from PIL import Image
client = QdrantClient(':memory:')
init_colab_collections(client)
img = Image.new('RGB', (224,224), color='green')
res = agentic_diagnose_colab(client, image=img, user_description='yellow spots', llm_func=lambda p: 'mocked')
print(res['prompt'])
```
