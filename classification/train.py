from itertools import chain
from typing import Iterable, Tuple

from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

from dataset_reader import ClassificationTsvReader
from model import SimpleClassifier


def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader()


def build_vocab(train_loader, dev_loader) -> Vocabulary:
    print("Building the Vocabulary")
    return Vocabulary.from_instances(
            chain(train_loader.iter_instances(), dev_loader.iter_instances())
            )


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
            )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


def build_data_loader(
        reader,
        train_data_path: str,
        validation_data_path: str,
        ) -> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
            reader, train_data_path, batch_size=8, shuffle=True, cuda_device=3
            )
    dev_loader = MultiProcessDataLoader(
            reader, validation_data_path, batch_size=8, shuffle=False, cuda_device=3
            )
    return train_loader, dev_loader


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        ) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)

    trainer = GradientDescentTrainer(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=train_loader,
            validation_data_loader=dev_loader,
            num_epochs=5,
            optimizer=optimizer,
            validation_metric="+accuracy",
            cuda_device=3,
            )
    return trainer


def run_training_loop(serialization_dir: str):
    reader = build_dataset_reader()
    train_loader, dev_loader = build_data_loader(
            reader,
            train_data_path='.data/imdb/aclImdb/train',
            validation_data_path='.data/imdb/aclImdb/test'
            )
    vocab = build_vocab(train_loader, dev_loader)
    model = build_model(vocab)
    model.to(3)

    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)

    print("Starting training")
    trainer.train()
    print("Finished training")


if __name__ == "__main__":
    run_training_loop("serialization_dir")
