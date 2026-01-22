# FarmFederate: AI-Powered Crop Stress Detection with Qdrant

**Convolve 4.0 - Pan-IIT AI/ML Hackathon | Qdrant Problem Statement**

> A multimodal AI system using Qdrant for **Search**, **Memory**, and **Recommendations** to address **Climate Resilience** and **Agricultural Sustainability** - a critical societal challenge.

---

## 1. Problem Statement

### What societal issue are we addressing?

**Global Food Security & Climate Resilience in Agriculture**

- **820 million people** face hunger globally (FAO, 2023)
- **Climate change** causes unpredictable crop stress patterns
- **Small-holder farmers** (70% of food production) lack access to AI diagnostics
- **Early detection gap**: Crop stress visible to AI 2-3 weeks before human detection
- **Data privacy concerns**: Farmers reluctant to share farm data with centralized systems

### Why does it matter?

- **30-40% yield loss** preventable with early stress detection
- **$220 billion** annual crop losses from pests and diseases worldwide
- **Privacy-preserving AI** enables trust and adoption among farmers
- **Multimodal understanding** (images + text observations) improves accuracy by 15-20%

**FarmFederate detects 5 crop stress types:**
| Stress Type | Detection Signals | Impact |
|-------------|------------------|--------|
| Water Stress | Wilting, leaf curl, soil dryness | 20-50% yield loss |
| Nutrient Deficiency | Yellowing, chlorosis, stunted growth | 15-40% yield loss |
| Pest Risk | Holes, webbing, insect damage | 10-30% yield loss |
| Disease Risk | Lesions, spots, fungal growth | 25-60% yield loss |
| Heat Stress | Scorching, browning, leaf burn | 15-35% yield loss |

---

## 2. System Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FARMFEDERATE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │ Flutter App  │────▶│           FastAPI Backend (server.py)            │  │
│  │  (Frontend)  │     │                                                  │  │
│  │              │     │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │  │
│  │ • Image      │     │  │  /predict  │  │   /rag     │  │  /memory   │  │  │
│  │ • Text Input │     │  │ Multimodal │  │  Search &  │  │  Session   │  │  │
│  │ • Dashboard  │     │  │ Classifier │  │ Recommend  │  │  History   │  │  │
│  └──────────────┘     │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │  │
│                       │        │               │               │         │  │
│                       └────────┼───────────────┼───────────────┼─────────┘  │
│                                │               │               │            │
│                                ▼               ▼               ▼            │
│                       ┌─────────────────────────────────────────────────┐   │
│                       │              QDRANT VECTOR DATABASE              │   │
│                       │                                                  │   │
│                       │  ┌─────────────────┐  ┌─────────────────┐       │   │
│                       │  │ crop_health_    │  │ farm_session_   │       │   │
│                       │  │ knowledge       │  │ memory          │       │   │
│                       │  │                 │  │                 │       │   │
│                       │  │ • visual: 512-d │  │ • semantic:     │       │   │
│                       │  │   (CLIP)        │  │   384-d         │       │   │
│                       │  │ • semantic:     │  │                 │       │   │
│                       │  │   384-d (SBERT) │  │ • farm_id       │       │   │
│                       │  │                 │  │ • timestamp     │       │   │
│                       │  │ Payload:        │  │ • diagnosis     │       │   │
│                       │  │ • stress_type   │  │ • treatment     │       │   │
│                       │  │ • crop_name     │  │ • feedback      │       │   │
│                       │  │ • severity      │  │                 │       │   │
│                       │  │ • source        │  │                 │       │   │
│                       │  └─────────────────┘  └─────────────────┘       │   │
│                       │                                                  │   │
│                       │  Distance: COSINE | Real-time | Low-latency     │   │
│                       └─────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    FEDERATED LEARNING LAYER                           │  │
│  │  • FedAvg aggregation across farms                                    │  │
│  │  • Differential privacy (ε, δ)-DP                                     │  │
│  │  • Non-IID data handling via Dirichlet distribution                   │  │
│  │  • 5 LLM + 5 ViT + 8 VLM models compared                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Qdrant is Critical to Our Solution

1. **Multimodal Named Vectors**: Qdrant's named vector support allows us to store both `visual` (512-d CLIP) and `semantic` (384-d SentenceTransformer) embeddings in the same point, enabling hybrid search.

2. **Metadata Filtering**: Qdrant's payload filtering enables queries like "find similar disease cases in tomato crops with severity > 3" - combining vector similarity with structured filters.

3. **Real-time Updates**: Farm data evolves constantly. Qdrant's real-time upsert allows us to update knowledge as new cases are diagnosed.

4. **Session Memory**: Qdrant persists farm session history with timestamps, enabling longitudinal analysis of crop health over growing seasons.

5. **Scalability**: As FarmFederate scales to thousands of farms, Qdrant's horizontal scalability ensures consistent low-latency retrieval.

---

## 3. Multimodal Strategy

### Data Types Used

| Data Type | Source | Embedding Model | Dimension |
|-----------|--------|-----------------|-----------|
| **Crop Images** | Camera/drone captures | CLIP (openai/clip-vit-base-patch32) | 512 |
| **Text Observations** | Farmer descriptions | SentenceTransformer (all-MiniLM-L6-v2) | 384 |
| **Sensor Data** | IoT devices | Encoded in payload metadata | - |
| **Historical Reports** | Agronomist notes | SentenceTransformer | 384 |

### How Embeddings are Created and Queried

```python
# Creating embeddings (from qdrant_rag.py)
class Embedders:
    def embed_image(self, image: Image) -> List[float]:
        """CLIP visual embedding (512-d, normalized)"""
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.clip.get_image_features(**inputs)
        vec = outputs.cpu().numpy()[0]
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_text(self, text: str) -> List[float]:
        """SentenceTransformer semantic embedding (384-d, normalized)"""
        vec = self.text_encoder.encode(text)
        return (vec / np.linalg.norm(vec)).tolist()

# Hybrid query (visual + semantic + filter)
results = client.search(
    collection_name="crop_health_knowledge",
    query_vector=("visual", image_embedding),
    query_filter=Filter(must=[
        FieldCondition(key="crop_name", match=MatchValue(value="tomato"))
    ]),
    limit=5,
    with_payload=True
)
```

---

## 4. Search / Memory / Recommendation Logic

### SEARCH: Semantic & Hybrid Retrieval

**Implementation**: `backend/qdrant_rag.py` → `agentic_diagnose()`

```python
def agentic_diagnose(client, image, user_description, top_k=3):
    """
    1. Embed query image with CLIP → visual vector
    2. Search Qdrant for top-k similar historical cases
    3. Build grounded prompt from retrieved payloads
    4. Generate evidence-based treatment plan
    """
    # Visual search
    img_vector = embedder.embed_image(image)
    hits = client.search(
        collection_name="crop_health_knowledge",
        query_vector=("visual", img_vector),
        limit=top_k,
        with_payload=True
    )

    # Build grounded response (no hallucination)
    retrieved_cases = [hit.payload for hit in hits]
    prompt = build_treatment_prompt(retrieved_cases, user_description)
    return {"retrieved": retrieved_cases, "treatment": llm_response(prompt)}
```

### MEMORY: Long-term Farm History

**Implementation**: `backend/farm_memory_agent.py` → `FarmMemoryAgent`

```python
class FarmMemoryAgent:
    """
    Persistent memory with:
    - farm_id scoping (multi-tenant)
    - Timestamp tracking (temporal queries)
    - Evolving representations (update on feedback)
    """

    def store_report(self, farm_id, diagnosis, treatment, image_embedding):
        """Store with semantic embedding for future retrieval"""
        payload = {
            "farm_id": farm_id,
            "timestamp": time.time(),
            "diagnosis": diagnosis,
            "treatment": treatment,
        }
        client.upsert(collection_name="farm_session_memory",
                      points=[PointStruct(id=uuid4(), vector={"semantic": embedding}, payload=payload)])

    def retrieve_history(self, farm_id, query=None, top_k=10):
        """Retrieve past diagnoses for this farm"""
        filter = Filter(must=[FieldCondition(key="farm_id", match=MatchValue(value=farm_id))])
        if query:
            return client.search(collection_name="farm_session_memory",
                                 query_vector=("semantic", embed_text(query)),
                                 query_filter=filter, limit=top_k)
        return client.scroll(collection_name="farm_session_memory",
                            scroll_filter=filter, limit=top_k)
```

### RECOMMENDATIONS: Context-Aware Treatment Plans

**Implementation**: Evidence-based recommendations grounded in retrieved data

```python
def get_treatment_recommendations(stress_type, severity, crop, retrieved_cases):
    """
    Traceable recommendations:
    1. Match stress_type to treatment database
    2. Rank by similarity to retrieved historical cases
    3. Return with evidence (which cases influenced decision)
    """
    recommendations = TREATMENT_DATABASE[stress_type]

    # Re-rank based on retrieved similar cases
    for case in retrieved_cases:
        if case["treatment_outcome"] == "success":
            boost_similar_treatments(recommendations, case)

    return [{
        "action": rec["action"],
        "priority": rec["priority"],
        "evidence": f"Based on {len(retrieved_cases)} similar cases",
        "source_cases": [c["id"] for c in retrieved_cases]  # Traceability
    } for rec in recommendations]
```

---

## 5. Limitations & Ethics

### Known Failure Modes

| Failure Mode | Mitigation |
|-------------|------------|
| **Novel stress patterns** not in training data | Uncertainty estimation flags low-confidence predictions |
| **Image quality issues** (blur, lighting) | Pre-processing validation; request re-capture |
| **Regional crop varieties** not represented | Federated learning incorporates local data over time |
| **Adversarial inputs** | Input validation; anomaly detection |

### Bias, Privacy, and Safety Considerations

1. **Data Bias**: Training data skewed toward commercial crops in temperate climates
   - **Mitigation**: Active collection from underrepresented regions; federated learning from diverse farms

2. **Privacy**: Farm data is sensitive (location, yields, practices)
   - **Mitigation**: Federated learning keeps raw data on-farm; only model updates shared
   - Differential privacy (ε=8.0, δ=1e-5) adds noise to aggregated updates

3. **Equity**: AI recommendations favor farms with connectivity
   - **Mitigation**: Offline-capable mobile app; SMS-based alerts

4. **Safety**: Incorrect pesticide recommendations could harm environment
   - **Mitigation**: Recommendations are suggestions, not prescriptions; always cite evidence

---

## 6. Quick Start

### Option A: Demo Mode (No Docker Required)

```bash
# Clone and setup
git clone https://github.com/your-repo/FarmFederate.git
cd FarmFederate

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install qdrant-client sentence-transformers

# Run with in-memory Qdrant
$env:DEMO_MODE='0'; $env:QDRANT_URL=':memory:'
python -m uvicorn backend.server:app --port 8000

# In another terminal - Frontend
cd frontend
flutter pub get
flutter run -d chrome
```

### Option B: Full Setup with Docker Qdrant

```bash
# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Backend with real Qdrant
$env:QDRANT_URL='http://localhost:6333'
pip install -r backend/requirements-qdrant.txt
python -m uvicorn backend.server:app --port 8000

# Frontend
cd frontend && flutter run -d chrome
```

### Option C: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/FarmFederate/blob/main/FarmFederate_Colab.py)

```python
# Quick smoke test
!python FarmFederate_Colab.py --auto-smoke --smoke-samples 50

# Full training with comparisons
!python FarmFederate_Colab.py --train --epochs 10 --max-samples 500 --use-qdrant
```

---

## 7. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Multimodal classification (image + text) |
| `/rag` | POST | RAG-based diagnosis with Qdrant retrieval |
| `/demo_populate` | POST | Populate Qdrant with demo vectors |
| `/demo_search` | POST | Search demo vectors (for testing) |
| `/telemetry` | POST | Receive IoT device telemetry |
| `/control/{device}` | POST | Control IoT devices (pump, heater, etc.) |

### Example: RAG Diagnosis

```bash
curl -X POST http://localhost:8000/rag \
  -F "image=@tomato_leaf.jpg" \
  -F "description=Yellow spots on lower leaves, spreading upward"
```

**Response:**
```json
{
  "retrieved": [
    {"id": 123, "score": 0.92, "payload": {"stress_type": "nutrient_def", "crop_name": "tomato"}},
    {"id": 456, "score": 0.87, "payload": {"stress_type": "disease_risk", "crop_name": "tomato"}}
  ],
  "treatment": {
    "likely_stress": ["nutrient_def", "disease_risk"],
    "recommendations": [
      {"action": "Apply balanced NPK fertilizer", "priority": "high", "evidence": "Based on 2 similar cases"}
    ],
    "traceability": "Retrieved cases 123, 456 from PlantVillage dataset"
  }
}
```

---

## 8. Model Comparison Results

### Performance Summary

| Model Type | Best Model | F1 Score | Parameters |
|------------|-----------|----------|------------|
| **LLM** | DistilBERT | 0.78 | 66M |
| **ViT** | DeiT-tiny | 0.81 | 5.7M |
| **VLM** | Attention Fusion | 0.85 | 72M |

### Federated vs Centralized

| Training Mode | LLM F1 | ViT F1 | VLM F1 |
|---------------|--------|--------|--------|
| Centralized | 0.78 | 0.82 | 0.86 |
| Federated | 0.75 | 0.79 | 0.83 |
| Gap | -0.03 | -0.03 | -0.03 |

*Federated learning achieves 96-97% of centralized performance while preserving privacy.*

---

## 9. Project Structure

```
FarmFederate/
├── backend/
│   ├── server.py              # FastAPI main application
│   ├── qdrant_rag.py          # Qdrant RAG utilities
│   ├── farm_memory_agent.py   # Session memory management
│   ├── multimodal_model.py    # RoBERTa + ViT classifier
│   ├── requirements.txt       # Base dependencies
│   └── requirements-qdrant.txt # Qdrant dependencies
├── frontend/
│   ├── lib/
│   │   ├── main.dart          # Flutter entry point
│   │   ├── services/api_service.dart  # API client
│   │   └── screens/chat_screen.dart   # Main UI
│   └── pubspec.yaml
├── FarmFederate_Colab.py      # Comprehensive Colab script
├── tests/
│   └── test_rag_endpoint.py   # RAG endpoint tests
└── README.md                  # This file
```

---

## 10. Resources

- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **CLIP Model**: https://huggingface.co/openai/clip-vit-base-patch32
- **SentenceTransformers**: https://www.sbert.net/
- **PlantVillage Dataset**: https://www.kaggle.com/datasets/emmarex/plantdisease

---

## License & Citation

MIT License

```bibtex
@software{farmfederate2026,
  title={FarmFederate: Federated Multimodal Learning for Crop Stress Detection with Qdrant},
  author={FarmFederate Team},
  year={2026},
  url={https://github.com/your-repo/FarmFederate},
  note={Convolve 4.0 - Qdrant Problem Statement}
}
```

---

**Version**: 3.0 (Qdrant + Comparisons Edition)
**Status**: Demo-ready for Convolve 4.0
