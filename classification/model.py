from typing import Dict

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # shape: (batch_size, emcoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(input=logits, target=label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

