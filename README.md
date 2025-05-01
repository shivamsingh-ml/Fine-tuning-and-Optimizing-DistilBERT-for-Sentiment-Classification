# Fine-tuning and Optimizing DistilBERT for Sentiment Classification

This project explores the fine-tuning of [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the SST-2 dataset for sentiment classification. It evaluates multiple classifier head architectures, fine-tuning techniques, and introduces infrastructure to support future integration of PEFT (e.g., LoRA, adapters).

**SST-2 (Stanford Sentiment Treebank v2)** is a binary sentiment classification dataset built from movie review phrases. It provides high-quality, phrase-level annotations that are ideal for benchmarking sentiment analysis.

**DistilBERT** is a distilled (compressed) version of BERT that retains 97% of language understanding capabilities while being 40% smaller and 60% faster. It is pretrained using knowledge distillation and is well-suited for downstream NLP tasks with limited compute resources.

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

| Experiment | Head Type | Hidden Layers       | Activation | Dropout | Trainable Params (%) | Validation Accuracy | Test Accuracy* |
|------------|-----------|---------------------|------------|---------|-----------------------|---------------------|----------------|
| Baseline   | HF Linear | None                | None       | None    | 100%                  | 91.51%              | 92.60%         |
| A          | MLP       | [256]               | ReLU       | 0.1     | 100%                  | 91.06%              |    -           |
| B          | MLP       | [512, 256]          | GELU       | 0.2     | 100%                  | 91.51%              |    -           |
| C          | MLP       | [512, 512, 256]     | SiLU       | 0.2     | 100%                  | 90.94%              |    -           |
| D          | MLP       | [768]               | ReLU       | 0.3     | 100%                  | 90.94%              |    -           |
| E          | MLP       | [1024, 512, 256]    | GELU       | 0.2     | 100%                  | 90.83%              |    -           |
| F          | MLP       | [256]               | SiLU       | 0.1     | 100%                  | 91.28%              |    -           |
| G          | MLP       | [512, 256, 128]     | GELU       | 0.15    | 100%                  | 90.71%              |    -           |

> *Test accuracy is reported via submission to the official [GLUE evaluation server](https://gluebenchmark.com/).

---

## 📒 Experiment Overview

- **Baseline**: Uses Hugging Face's built-in linear head without any architectural modifications. Serves as the control for comparison.
- **Exp A**: Lightweight MLP with a single 256-unit hidden layer and ReLU activation. Tests minimal non-linearity.
- **Exp B**: Deeper head with two layers and GELU activation, investigating performance from added depth.
- **Exp C**: Very deep MLP with SiLU activation, aimed at capturing complex non-linearities.
- **Exp D**: Wide single-layer model with increased dropout to test regularization in larger capacity heads.
- **Exp E**: Very wide and deep head; useful for examining overfitting behavior on small datasets.
- **Exp F**: Modern shallow configuration with SiLU; combines simplicity with advanced activation.
- **Exp G**: Deep head with progressive bottleneck structure, common in deep residual networks.

---

## 🧠 PEFT vs Full Fine-tuning Comparison

In addition to full fine-tuning experiments, Parameter-Efficient Fine-Tuning (PEFT) methods are explored for improving efficiency while maintaining competitive performance. The table below summarizes their comparison with the baseline full fine-tuning approach.

| Method | Trainable Params (%) | Validation Accuracy | Test Accuracy* |
|--------|----------------------|---------------------|----------------|
| Full Fine-tuning (Baseline) | 100% | 91.51% | 92.60% |
| LoRA | ~1.8% | (to fill) | (to fill) |
| QLoRA (Planned) | ~0.3% | (to fill) | (to fill) |
| Adapters (Planned) | ~1–3% | (to fill) | (to fill) |

> *Test accuracy is reported via submission to the official [GLUE evaluation server](https://gluebenchmark.com/).

---

### 📒 PEFT Method Overview

- **Full Fine-tuning (Baseline)**:  
  The entire DistilBERT model was fine-tuned along with the classification head. This serves as the control and reference point for PEFT methods.

- **LoRA**:  
  Low-Rank Adapters were added to key transformer layers (`q_lin`, `k_lin`, `v_lin`, `out_lin`, `lin1`, `lin2`). Only ~1.8% of total parameters were trainable, offering significant savings in memory and computation while achieving near full fine-tuning performance.

- **QLoRA (Planned)**:  
  Combines 4-bit quantization with LoRA adapters. Expected to offer even lower memory usage while maintaining LoRA's efficiency. Will be explored in upcoming experiments.

- **Adapters (Planned)**:  
  Lightweight adapters inserted into transformer layers as an alternative PEFT approach. They allow selective fine-tuning with small trainable modules and are often used for rapid adaptation across tasks.

## 📊 Future Plans

- Extend to GLUE/SuperGLUE tasks beyond SST-2
- Improve tokenizer flexibility & model checkpointing