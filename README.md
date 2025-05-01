
# Fine-tuning and Optimizing DistilBERT for Sentiment Classification

This project explores the fine-tuning of [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the SST-2 dataset for sentiment classification. It evaluates multiple classifier head architectures, fine-tuning techniques, and introduces infrastructure to support future integration of PEFT (e.g., LoRA, adapters).

**SST-2 (Stanford Sentiment Treebank v2)** is a binary sentiment classification dataset built from movie review phrases. It provides high-quality, phrase-level annotations ideal for benchmarking sentiment analysis.

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

## 🔧 Setup

```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

```bash
# Run baseline experiment
python experiments/run_experiment.py --config config/baseline.yaml

# Run experiment with custom classifier head (example: A)
python experiments/run_experiment.py --config config/exp_a.yaml
```

Each run will save results (`results.txt`) and log optionally to [Weights & Biases](https://wandb.ai).

---

## 🔮 Techniques Used

- **Classifier Heads**: Linear and MLP heads with varied depth and activations (ReLU, GELU, SiLU)
- **Training Optimizations**:
  - Mixed precision (FP16)
  - Cosine LR scheduling, Weight decay, Warmup
- **Evaluation**: Using `evaluate` library (GLUE SST-2 accuracy)
- **Reproducibility**: Fixed seeds and deterministic dataloaders
- **Logging**: Weights & Biases (W&B) and local logs
- **Extensibility**: Modular code for easy PEFT integration and domain adaptation experiments

---

## 📊 Experiment Results (SST-2)

| Experiment | Head Type | Hidden Layers | Activation | Dropout | Trainable Params (%) | Validation Accuracy | Test Accuracy* |
|------------|-----------|---------------|------------|---------|---------------------|---------------------|----------------|
| Baseline   | HF Linear | None          | None       | None    | 100%                 | 91.51%               | 92.60%         |
| A          | MLP       | [256]         | ReLU       | 0.1     | 100%                 | 91.06%               | -              |
| B          | MLP       | [512, 256]    | GELU       | 0.2     | 100%                 | 91.51%               | -              |
| C          | MLP       | [512, 512, 256] | SiLU     | 0.2     | 100%                 | 90.94%               | -              |
| D          | MLP       | [768]         | ReLU       | 0.3     | 100%                 | 90.94%               | -              |
| E          | MLP       | [1024, 512, 256] | GELU    | 0.2     | 100%                 | 90.83%               | -              |
| F          | MLP       | [256]         | SiLU       | 0.1     | 100%                 | 91.28%               | -              |
| G          | MLP       | [512, 256, 128] | GELU     | 0.15    | 100%                 | 90.71%               | -              |

> *Test accuracy via GLUE official evaluation server.

---

## 🧠 PEFT vs Full Fine-tuning Comparison

| Method               | Trainable Params (%) | Validation Accuracy | Test Accuracy* |
|----------------------|----------------------|---------------------|----------------|
| Full Fine-tuning (Baseline) | 100%           | 91.51%               | 92.60%         |
| LoRA                 | ~1.8%                 | 90.25%               | -              |
| QLoRA                | ~1.8%                 | 90.25%               | -              |

### 📒 PEFT Overview

- **LoRA**: Low-Rank Adapters inserted into key transformer layers. Achieved ~91% accuracy using only ~1.8% trainable parameters.
- **QLoRA**: Combines LoRA with 4-bit quantization for efficiency and was successfully tested.

- **Baseline**: Full fine-tuning of all model parameters.

---

## 🌎 Domain Adaptation (SST-2 → IMDb)

| Model | IMDb Accuracy | Notes |
|-------|---------------|-------|
| SST-2 Fine-tuned (Zero-shot on IMDb) | 87.34% | Without IMDb finetuning |
| IMDb Domain Adapted (SST-2 + IMDb Fine-tuned) | 91.42% | Finetuned on IMDb |

> **Domain Adaptation Gain** → **+4.08%**

### 📒 Domain Adaptation Overview

- **Zero-shot (SST-2 Model)**:  
  Direct transfer to IMDb resulted in good accuracy, proving generalization.
  
- **IMDb Domain Adaptation**:  
  Further finetuning specialized the model for IMDb movie reviews, boosting accuracy by ~4%.

- **Conclusion**:  
  Domain adaptation successfully improved performance while leveraging existing SST-2 sentiment knowledge.

---

## 📊 Future Plans

- Explore PEFT in-depth (Adapters and QLoRA optimizations)
- Extend pipeline to other GLUE/SuperGLUE tasks
- Improve tokenizer flexibility and advanced checkpointing
- Add broader real-world domain transfer tasks
