# Backend package for FarmFederate
# Perform guarded imports so that optional heavy dependencies (qdrant-client, transformers, etc.)
# do not cause import-time failures for lightweight unit tests.
try:
    from .qdrant_utils import get_qdrant_client, initialize_crop_health_collection, initialize_farm_session_collection
except Exception:
    get_qdrant_client = None
    initialize_crop_health_collection = None
    initialize_farm_session_collection = None

try:
    from .ingest_qdrant import ingest_datasets, Embedders
except Exception:
    ingest_datasets = None
    Embedders = None

try:
    from .agent_rag import rag_diagnose, store_session_memory, check_session_history
except Exception:
    rag_diagnose = None
    store_session_memory = None
    check_session_history = None

# New single-file utilities are available separately: backend.qdrant_rag
