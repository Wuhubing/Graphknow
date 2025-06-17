import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "tokenizer_name": "meta-llama/Llama-2-7b-hf",
    "load_in_4bit": True,
    "device_map": "auto",
}

EXPERIMENT_CONFIG = {
    "dataset": "hotpot_qa",
    "dataset_split": "validation",
    "sample_size": 100,
    "max_new_tokens": 50,
    "batch_size": 1,
    "temperature": 0.1,  # Low temperature for more deterministic outputs
    "do_sample": False,  # Greedy decoding for consistency
}

GRAPH_CONFIG = {
    "entity_extractor": "en_core_web_sm",
    "max_distance": 5,
    "edge_threshold": 1,
}

PERTURBATION_STRATEGIES = [
    {"name": "high_degree", "desc": "High degree nodes"},
    {"name": "close_distance", "desc": "Distance 1 from core path"},
    {"name": "far_distance", "desc": "Distance 3+ from core path"},
    {"name": "high_centrality", "desc": "High betweenness centrality"},
]