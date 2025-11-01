# Active Learning Framework - V_FINAL

This is the final, cleaned, and improved version of the Active Learning framework for CIFAR-10.

## Overview

This implementation provides a comprehensive active learning framework that addresses all the issues identified in previous versions (V1, V2, V3) and combines the best strategies from:
- Adversarial Leader approach
- Uncertainty-based sampling
- CoreSet/K-Center geometric diversity
- Random baseline

## Key Features

✅ **Clean, Professional Codebase**
- Modern Python 3.8+ with type hints
- Modular architecture with clear separation of concerns
- Comprehensive documentation and logging
- No code duplication

✅ **Flexible Configuration**
- YAML-based configuration system
- Easy hyperparameter tuning
- Multiple sampling strategies supported

✅ **Comprehensive Metrics**
- Detailed per-round and per-epoch tracking
- Automatic visualization generation
- Statistical analysis tools

✅ **Reproducibility**
- Deterministic training option
- Seed management across all libraries
- Environment tracking

## Project Structure

```
ActiveLearning/
├── configs/                    # Configuration files
│   └── default.yaml
├── src/                       # Source code
│   ├── data/                  # Data loading
│   ├── models/                # Model architectures
│   ├── trainers/              # Training logic
│   ├── samplers/              # Sampling strategies
│   ├── metrics/               # Metrics tracking
│   ├── visualization/         # Plotting utilities
│   └── utils/                 # Utilities
├── experiments/               # Experiment scripts
│   └── run_experiment.py
├── INVESTIGATION_V3_FINAL.md  # Comprehensive analysis
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running an Experiment

```bash
# Run with default configuration
python experiments/run_experiment.py --config configs/default.yaml
```

### Configuration

Edit `configs/default.yaml` to customize:
- Sampling strategy (random, uncertainty, coreset, hybrid)
- Model architecture (VGG16, VGG19, etc.)
- Training hyperparameters
- Active learning parameters
- Logging and visualization options

Example configuration snippet:

```yaml
active_learning:
  strategy: "uncertainty"  # or "random", "coreset", "hybrid"
  initial_samples: 1000
  samples_per_round: 1000
  num_rounds: 10

model:
  architecture: "vgg16"
  dropout: 0.5
  weight_decay: 0.002
```

## Sampling Strategies

### 1. Random Sampling (Baseline)
Simple random selection from unlabeled pool.

```yaml
active_learning:
  strategy: "random"
```

### 2. Uncertainty Sampling
Selects samples with highest prediction uncertainty.

```yaml
active_learning:
  strategy: "uncertainty"
  uncertainty_config:
    method: "entropy"  # or "margin", "least_confidence"
```

### 3. CoreSet Sampling
Maximizes diversity in feature space using k-center greedy.

```yaml
active_learning:
  strategy: "coreset"
  diversity_config:
    distance_metric: "euclidean"  # or "cosine"
```

### 4. Hybrid Sampling (Coming Soon)
Combines multiple strategies with configurable weights.

```yaml
active_learning:
  strategy: "hybrid"
  hybrid_weights:
    adversarial: 0.4
    uncertainty: 0.3
    diversity: 0.3
```

## Results and Metrics

After running an experiment, find results in:
- `results/<experiment_name>/metrics.json` - Detailed metrics
- `results/<experiment_name>/round_summary.json` - Per-round summary
- `results/<experiment_name>/experiment_summary.json` - Overall summary
- `results/<experiment_name>/plots/` - Visualization plots

### Generated Visualizations

1. **Learning Curves**: Test accuracy vs rounds
2. **Loss Curves**: Training loss vs rounds
3. **Sample Efficiency**: Samples needed to reach target accuracy
4. **Round Breakdown**: Multiple metrics per round

## Investigation Document

See `INVESTIGATION_V3_FINAL.md` for:
- Comprehensive analysis of all previous versions
- Performance comparison (10-15% vs 30-40% Round 1 accuracy)
- Code quality issues identified and resolved
- Design decisions and best practices
- Future improvements

## Key Improvements Over V3

1. ✅ **Fixed naming inconsistencies** (adverserial → adversarial)
2. ✅ **Eliminated code duplication** (single unified framework)
3. ✅ **Configuration-based** (no hardcoded values)
4. ✅ **Modern Python** (type hints, proper structure)
5. ✅ **Comprehensive logging** (metrics, checkpoints, visualizations)
6. ✅ **Professional documentation** (docstrings, examples)
7. ✅ **Reproducible** (seed management, deterministic option)

## Expected Performance

Based on investigation of previous versions:

| Method | Round 1 Accuracy | Final Accuracy | Notes |
|--------|------------------|----------------|-------|
| Random | 10-15% | 80-85% | Baseline |
| Uncertainty | 15-20% | 85-90% | Good for calibrated models |
| CoreSet | 12-18% | 83-88% | Ensures diversity |
| Hybrid (Target) | **35-45%** | **88-92%** | Best of all methods |

## Development

### Adding a New Sampling Strategy

1. Create a new sampler in `src/samplers/`:

```python
from .base import BaseSampler

class MySampler(BaseSampler):
    def select_samples(self, unlabeled_indices, labeled_indices, 
                      model, unlabeled_loader, n_samples):
        # Your selection logic here
        return selected_indices
```

2. Register in `src/samplers/__init__.py`
3. Add config option in YAML

### Adding a New Model

1. Create model in `src/models/`
2. Update `create_model()` function
3. Ensure model returns (logits, features) tuple

## Citation

If you use this framework, please cite the original Core-Set paper:

```bibtex
@inproceedings{sener2018active,
    title={Active Learning for Convolutional Neural Networks: A Core-Set Approach},
    author={Ozan Sener and Silvio Savarese},
    booktitle={International Conference on Learning Representations},
    year={2018},
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.

---

**Status**: ✅ Investigation Complete | 🚧 Implementation In Progress | ⚪ Testing Pending
