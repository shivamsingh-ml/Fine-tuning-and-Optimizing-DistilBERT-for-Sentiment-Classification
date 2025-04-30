import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from src.models.custom_heads import MLPHead, LinearHead

class DistilBERTWithCustomHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_type = config["model"]["classifier_head"]["type"]

        if self.head_type == "pretrained":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["model"]["pretrained_model_name"],
                num_labels=2
            )
        else:
            self.encoder = AutoModel.from_pretrained(config["model"]["pretrained_model_name"])
            hidden_size = self.encoder.config.hidden_size
            head_cfg = config["model"]["classifier_head"]

            if self.head_type == "linear":
                self.classifier = LinearHead(
                    input_dim=hidden_size,
                    dropout=head_cfg.get("dropout", 0.1)
                )
            elif self.head_type == "mlp":
                self.classifier = MLPHead(
                    input_dim=hidden_size,
                    hidden_layers=head_cfg["hidden_layers"],
                    activation=head_cfg["activation"],
                    dropout=head_cfg["dropout"]
                )
            else:
                raise ValueError(f"Unsupported head type: {self.head_type}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.head_type == "pretrained":
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Custom head path
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )
