# Memory Integration — Qdrant + FarmMemoryAgent 🔧

This document explains the Qdrant-based long-term memory integration for FarmFederate-Advisor.
It describes how the Flamingo VLM outputs are stored in Qdrant, how retrieval works, and the ethical/limitations section required for Convolve 4.0.

---

## 1) Architecture Overview 🏗️

- Flamingo VLM / SensorAwareVLM: produces two embeddings when generating a `Crop Stress Analysis Report`:
  - **visual**: ViT/CLIP-style image embedding (e.g., 512-d)
  - **semantic**: LLM-style textual embedding (e.g., 384-d)
- FarmMemoryAgent: wraps Qdrant to persist the pair of named vectors into a `farm_history` collection.
- Query flow:
  1. A new image arrives; VLM produces `visual` embedding and an on-the-fly `semantic` summary embedding (if available).
  2. FarmMemoryAgent.retrieve_similar_by_image(visual, farm_id) runs a visual nearest-neighbor search restricted with a `farm_id` filter.
  3. Top-K hits are returned with payloads (report text, timestamp, metadata) and used as context for the next diagnosis.

This design separates short-lived prompt context (single prompt) from long-term memory (per-farm historical events).

---

## 2) Capability Mapping (for Convolve 4.0) ✅

| Capability | Implementation details | Why it satisfies Convolve 4.0 |
|---|---|---|
| Multimodal Retrieval | Store both **visual** (ViT) and **semantic** (LLM) embeddings as named vectors in Qdrant. | Demonstrates correct use of vector embeddings and cross-modal retrieval capabilities. |
| Long-term Memory | Per-farm `farm_id` payload filter used to scope retrieval to one farm's history. | Memory beyond a single prompt — recall past stress events from the same farm (e.g., "Severe Water Stress on Day 5"). |
| Traceable Reasoning | Each point stores text + timestamp + metadata; retrieval includes payload that can be logged for evidence. | Enables evidence-based recommendations with an audit trail (which historical case informed the output). |

---

## 3) Search / Memory / Recommendation Logic (short)

1. When storing a new analysis, save:
   - visual vector (`visual`), semantic vector (`semantic`), `farm_id`, `report` text, `timestamp`, and `meta` (labels, severity).
2. For a new query image: compute `visual` embedding -> run Qdrant search with `vector_name='visual'` and `Filter(must: Match(farm_id=<id>))` -> get top 3 results.
3. Combine the retrieved textual payloads with the current prompt/context for the VLM/LLM to produce a traceable, evidence-based diagnosis.

This scoping ensures the memory is relevant to the specific farm (localization of context), which improves recommendation precision and maintains privacy boundaries.

---

## 4) Exporting Interaction Logs (CSV / Sheets) 📤

- The agent includes `export_reports_to_csv(farm_id, out_path)` to save the farm history as a CSV.
- To push to Google Sheets, you can use `gspread` to upload the CSV or write rows directly with the API.

Example (conceptual):

```python
# export to csv then upload to Sheets (requires credentials + gspread)
csv_path = agent.export_reports_to_csv('farm-A', out_path='farm-A-history.csv')
# use gspread to push csv to a sheet (not included by default - optional)
```

---

## 5) Limitations & Ethical Considerations ⚖️

- **Privacy**: Federated learning keeps raw images at the edge when possible; only embeddings or synthesized reports are centralized. Still, metadata (farm_id, timestamps) can be sensitive — treat with access controls and encryption at rest.
- **Bias**: The model depends on diverse datasets (PlantVillage, IP102, etc.). Without geographically diverse samples, the model may underperform on region-specific crops or pests.
- **Failure Modes**: Low-light, motion blur, or occlusions are common field issues that degrade ViT encoder performance. The system should log model confidence and revert to human-in-the-loop for low-confidence or high-stakes recommendations.

---

## 6) Sample Interaction Logs (Evidence)

- Store example reports (retain three successful sample reports from current run as evidence). Include them in the appendix of your 10-page report.

---

## 7) Reproducibility Checklist 🔁

- requirements-qdrant.txt (contains qdrant-client, numpy, pandas)
- The `backend/farm_memory_agent.py` file provides a reproducible API for storing and retrieving.
- Add a short runbook showing how to start an in-memory Qdrant demo for reviewers.

---

If you'd like, I can: 
- Draft the 1-page architecture overview slide for your report, or
- Add a small demo notebook cell that stores a few sample reports and demonstrates retrieval for one `farm_id`.

A runnable demo script and CI smoke workflow have been added:
- `backend/demo_farm_memory_demo.py` runs a small in-memory Qdrant demo (stores 3 records, retrieves top-3, and exports a CSV).
- `.github/workflows/qdrant-memory-smoke.yml` executes the demo and the FarmMemoryAgent unit test on pushes/PRs.

Would you like the architecture slide or the demo notebook cell added next? 💾