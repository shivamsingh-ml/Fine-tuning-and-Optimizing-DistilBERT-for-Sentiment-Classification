# Fine-tuning and Optimizing DistilBERT for Sentiment Classification

This project explores the fine-tuning of [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the SST-2 dataset for sentiment classification. It evaluates multiple classifier head architectures, fine-tuning techniques, and introduces infrastructure to support future integration of PEFT (e.g., LoRA, adapters).

---

## 📂 Project Structure

```bash
.
├── config/                  # YAML configs for each experiment
├── experiments/
│   └── run_experiment.py   # Main training script
├── src/
│   ├── data.py             # Tokenization and dataset loading
│   ├── utils.py            # Seeding, metrics
│   └── models/
│       ├── custom_heads.py # Linear and MLP classifier heads
│       └── wrapped_model.py# Combines encoder + head
├── results/                # Training outputs & results.txt per experiment
├── requirements.txt
└── README.md
```

---

## 🔢 Setup

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

```bash
# Baseline
python experiments/run_experiment.py --config config/baseline.yaml

# Custom classifier head (example: A)
python experiments/run_experiment.py --config config/exp_a.yaml
```

Each run will save a `results.txt` in the specified `output_dir`, and optionally log to [Weights & Biases](https://wandb.ai).

---

## 🔮 Techniques Used

- 🔀 **Classifier heads**: Linear and MLP heads with variable depth, width, and activations (ReLU, GELU, SiLU)
- 🔧 **Training optimizations**:
  - Mixed precision (FP16)
  - Cosine LR scheduling
  - Weight decay
  - Warmup steps
- 📊 **Metrics**: Evaluation using `evaluate` (GLUE SST-2 accuracy)
- 💼 **Reproducibility**: Global seed setup, deterministic dataloading
- 📊 **Logging**: Optional Weights & Biases integration per run
- 🚪 **Extensible**: Supports custom heads now, PEFT in the future

---

## 📊 Experiment Results

| Experiment | Head Type | Hidden Layers       | Activation | Dropout | PEFT    | Trainable Params (%) | Validation Accuracy | Test Accuracy* | Notes                                |
|------------|-----------|---------------------|------------|---------|---------|-----------------------|---------------------|----------------|--------------------------------------|
| Baseline   | HF Linear | None                | None       | None    | None    | 100%                  | 91.51%              | 92.60%         | HuggingFace pretrained head          |
| A          | MLP       | [256]               | ReLU       | 0.1     | None    | 100%                  |                     |                | Light MLP head                       |
| B          | MLP       | [512, 256]          | GELU       | 0.2     | None    | 100%                  | 91.16%              |                | Deeper head                          |
| C          | MLP       | [512, 512, 256]     | SiLU       | 0.2     | None    | 100%                  |                     |                | Very deep MLP                        |
| D          | MLP       | [768]               | ReLU       | 0.3     | None    | 100%                  |                     |                | Wide single layer                    |
| E          | MLP       | [1024, 512, 256]    | GELU       | 0.2     | None    | 100%                  |                     |                | Wide and deep, possible overfit      |
| F          | MLP       | [256]               | SiLU       | 0.1     | None    | 100%                  |                     |                | Shallow, modern activation           |
| G          | MLP       | [512, 256, 128]     | GELU       | 0.15    | None    | 100%                  |                     |                | Progressive bottleneck               |
| PEFT-A     | Linear    | None                | None       | 0.1     | LoRA    | ~20%                  |                     |                | PEFT w/ LoRA on baseline             |
| PEFT-B     | MLP       | [256]               | GELU       | 0.1     | Adapter | ~15%                  |                     |                | Adapter-based fine-tuning            |

> *Test accuracy is reported via submission to the official [GLUE evaluation server](https://gluebenchmark.com/).

---

## 📊 Future Plans

- Integrate PEFT methods (LoRA, QLoRA, Adapters)
- Add results aggregation script for summary.csv generation
- Extend to GLUE/SuperGLUE tasks beyond SST-2
- Improve tokenizer flexibility & model checkpointing
```

Let me know if you'd like this exported to `.md` or if you want a `requirements.txt` template added as well.

