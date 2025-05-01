import argparse
import os
import yaml
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import wandb
from src.utils import set_global_seed, compute_metrics
from src.data import load_and_preprocess_dataset
from src.models.wrapped_model import DistilBERTWithCustomHead


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config["experiment"]["seed"])

    if config["training"].get("report_to") == "wandb":
        wandb.init(
            project=config["training"]["wandb_project"],
            name=config['experiment']['name'],
            config=config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model_name"])
    dataset = load_and_preprocess_dataset(tokenizer, config["data"]["max_length"])

    model = DistilBERTWithCustomHead(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "linear"),
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"].get("greater_is_better", True),
        fp16=config["training"]["fp16"],
        logging_dir=config["training"]["logging_dir"],
        logging_strategy=config["training"]["logging_strategy"],
        logging_steps=config["training"]["logging_steps"],
        report_to=config["training"].get("report_to", "none"),
        seed=config["experiment"]["seed"],
        data_seed=config["experiment"]["seed"],
        run_name=config['experiment']['name']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    val_acc = eval_results.get("eval_accuracy", 0.0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    results_file = os.path.join(config["training"]["output_dir"], "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Experiment: {config['experiment']['name']}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write("Config used:\n")
        yaml.dump(config, f)
    save_dir = os.path.join(config["training"]["output_dir"], "final_model")
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model and tokenizer saved to {save_dir}")

if __name__ == "__main__":
    main()