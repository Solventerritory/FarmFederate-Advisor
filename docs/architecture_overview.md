# Architecture Overview — FarmFederate Memory Layer (1-page)

This one-page overview explains how the Flamingo VLM and Qdrant-based FarmMemoryAgent work together to provide "Memory Beyond a Single Prompt".

## Components
- Flamingo VLM / SensorAwareVLM
  - Inputs: image + optional sensor text
  - Outputs: diagnosis text + image embedding (`visual`) + optional text embedding (`semantic`)

- FarmMemoryAgent (Qdrant)
  - Stores records in collection `farm_history` with named vectors:
    - `visual` (ViT/CLIP 512-d) — used for visual similarity searches
    - `semantic` (LLM/text 384-d) — optional, for semantic nearest-neighbor
  - Payload: `farm_id`, `report`, `timestamp`, `meta` (e.g., label, severity)

- Query Flow
  1. New image arrives → VLM computes `visual` embedding.
  2. Call `retrieve_similar_by_image(visual, farm_id, top_k=3)` → Qdrant returns top-K similar historical events for that farm.
  3. Aggregated payloads (reports + timestamps) are used as context evidence for the VLM/LLM or for human review.

## Why this satisfies the Convolve 4.0 Memory requirement
- Long-term memory is scoped per `farm_id`, enabling history recall across days (not limited to a single prompt).
- Named vectors enable true multimodal retrieval (visual + semantic).
- Payload storage provides traceability for evidence-based recommendations.

---

If you want a formal slide (PNG/SVG) I can generate a small diagram image and add it to `docs/` for the final report.
