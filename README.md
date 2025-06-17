# Knowledge Ripple Effect in Large Language Models

This repository contains the implementation and analysis of the "Knowledge Ripple Effect" experiment, which investigates how knowledge influences knowledge in large language models.

## Key Finding: Semantic Similarity Paradox

Our most important discovery is that **semantically similar knowledge reinforces rather than disrupts existing knowledge** (p=0.0007, statistically significant). This counterintuitive finding has profound implications for understanding LLM robustness.

## Project Structure

```
knowledge_ripple_effect/
├── src/                          # Core modules
│   ├── config.py                 # Configuration and experiment parameters
│   ├── model_utils.py            # LLM inference utilities
│   ├── data_processing.py        # HotpotQA data processing
│   ├── knowledge_graph.py        # Knowledge graph construction
│   ├── baseline_evaluation.py    # Baseline model evaluation
│   ├── perturbation_experiment.py # Perturbation experiment logic
│   ├── semantic_utils.py         # Semantic similarity computation
│   ├── ripple_analyzer.py        # Statistical analysis and visualization
│   ├── analysis_visualization.py # Alternative visualization tools
│   └── utils.py                  # Utility functions
│
├── run_complete_experiment.py    # Main experiment runner
├── run_semantic_experiment.py    # Semantic-focused experiments
│
├── analysis_tools/               # Analysis and visualization scripts
│   ├── analyze_bright_knowledge.py
│   ├── analyze_perturbation_patterns.py
│   ├── semantic_pattern_analysis.py
│   ├── semantic_similarity_deep_analysis.py
│   ├── generate_paper_tables.py
│   ├── case_comparison_visualizer.py
│   ├── detailed_case_analysis.py
│   ├── qualitative_case_analysis.py
│   └── combine_experiment_results.py
│
└── results/                      # Experiment results (gitignored)
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Experiment

### 1. Complete Experiment Pipeline

```bash
python run_complete_experiment.py
```

This runs the full pipeline:
- Baseline evaluation on HotpotQA samples
- Knowledge graph construction
- Perturbation experiments with multiple strategies
- Statistical analysis and visualization

### 2. Semantic Similarity Focused Experiment

```bash
python run_semantic_experiment.py
```

This focuses specifically on semantic similarity effects.

## Core Components

### Perturbation Strategies

1. **Graph Structure Based**:
   - `high_degree`: Target nodes with many connections
   - `close_distance`: Target nodes 1 hop from core
   - `far_distance`: Target nodes 3+ hops from core
   - `high_centrality`: Target nodes with high betweenness centrality

2. **Semantic Similarity Based**:
   - `high_semantic_similarity`: Similar knowledge (>0.7 similarity)
   - `medium_semantic_similarity`: Moderate similarity (0.3-0.7)
   - `low_semantic_similarity`: Dissimilar knowledge (<0.3)

### Key Findings

| Strategy | Flip Rate | Avg Confidence Change |
|----------|-----------|----------------------|
| high_degree | 22.22% | -0.0018 |
| low_semantic_similarity | 21.11% | -0.0167 |
| high_semantic_similarity | 6.17% | -0.0231 |

### Analysis Tools

After running experiments, use these tools for deeper analysis:

```bash
# Analyze patterns in perturbation results
python analyze_perturbation_patterns.py

# Generate visualizations for case comparisons
python case_comparison_visualizer.py

# Extract semantic patterns from successful flips
python semantic_pattern_analysis.py

# Generate LaTeX tables for paper
python generate_paper_tables.py
```

## Results

Experiment results are saved in `results/` with timestamps:
- `baseline/`: Baseline evaluation results
- `perturbation/`: Perturbation experiment results
- `analysis/`: Statistical analysis and visualizations

## Configuration

Edit `src/config.py` to modify:
- Model settings (default: Llama-2-7b with 4-bit quantization)
- Experiment parameters (sample size, temperature, etc.)
- Perturbation strategies

## Hardware Requirements

- GPU: NVIDIA A40 (40GB) or similar
- RAM: 32GB+ recommended
- Storage: ~10GB for model and data

## Citation

If you use this code in your research, please cite:

```bibtex
@article{knowledge-ripple-effect,
  title={Knowledge Ripple Effect: How Knowledge Influences Knowledge in Large Language Models},
  author={[Your Name]},
  year={2024}
}
```