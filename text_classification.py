import torch
import tempfile
from itertools import islice
from typing import Iterable, Dict, List, Tuple, Optional
from allennlp.common import JsonDict
from allennlp.nn.util import get_text_field_mask
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Instance, DatasetReader, TextFieldTensors, DataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp_tutorial.JiebaTokenizer.jiebatokenizer import JiebaTokenzer
from allennlp_tutorial.BertTokenizer.berttokenizer import BERTTokenizer
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.vocabulary import Vocabulary, _read_pretrained_tokens
from allennlp.data.token_indexers import TokenIndexer, SpacyTokenIndexer, SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.util import evaluate
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.predictors import Predictor
import warnings
warnings.filterwarnings("ignore")


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self, lazy: bool = False,
                       tokenizer: Tokenizer = None,
                       token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        text_field = TextField(tokens, self._token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in islice(lines, 1, 100):
                text, _, __, label = line.strip().split('\t')
                yield self.text_to_instance(text, label)

    # def _read(self, file_path: str) -> Iterator[Instance]:
    #     """使用pandas读取文件"""
    #     import pandas as pd
    #     df = pd.read_csv(file_path, sep='\t')
    #     # if config.testing: df = df.head(1000)
    #     for i, row in df.iterrows():
    #         yield self.text_to_instance(text=self._tokenizer.tokenize(row["text"]),
    #                                     label=row["label"])


def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    # training_data = reader.read("quick_start/data/movie_review/train.tsv")
    # validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
    return training_data, validation_data


@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                       embedder: TextFieldEmbedder,
                       encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: TextFieldTensors,
                      label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1) # 如果只需要计算loss，这步可以不需要，因为torch的CE会用log_softmax处理好
        outputs = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            # Shape: (1,)
            outputs['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


def build_vocab(instances: Iterable[Instance],
                pretrained_files: Optional[Dict[str, str]] = None,
                include_full_pretrained_words: bool = False
                ) -> Vocabulary:
    print("Building the vocabulary")
    vocab = Vocabulary.from_instances(instances)
    if pretrained_files and include_full_pretrained_words:
        pretrained_tokens = _read_pretrained_tokens(pretrained_files["tokens"])
        from collections import Counter
        c = Counter(pretrained_tokens)
        counter = {"tokens": dict(c)}
        vocab._extend(counter=counter)
    print("Vocab size: ", vocab.get_vocab_size("tokens"))
    return vocab


def build_model(vocab: Vocabulary, embedding_dim: int, pretrained_file: str = None) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, pretrained_file=pretrained_file, vocab=vocab)
    embedder = BasicTextFieldEmbedder({"tokens": embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim)
    return SimpleClassifier(vocab, embedder, encoder)


def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset = None,
    batch_size: int = 8):
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    batch_sampler = BucketBatchSampler(train_data, batch_size=batch_size, sorting_keys=["text"], padding_noise=0)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler)
    # dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)
    return train_loader #, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader = None,
    num_epochs: int = 1,
    cuda_device: int = -1
    ) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        cuda_device=cuda_device
    )
    return trainer


def run_training_loop():
    tokenizer = BERTTokenizer(vocab_file='/Users/tianhongzxy/Downloads/BiSentESIM/BiSentESIM/My-pipeline/allennlp_tutorial/BertTokenizer/vocab.txt')
    # tokenizer = BERTTokenizer('bert-base-multilingual-cased')
    dataset_reader = ClassificationTsvReader(tokenizer=tokenizer)
    print("Reading data")
    train_data = dataset_reader.read('/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.txt')
    pretrained_files = {"tokens": "/Users/tianhongzxy/Downloads/BiSentESIM/BiSentESIM/embedding/glove.6B.300d.txt"}
    cuda_device = -1
    batch_size = 8
    vocab = build_vocab(train_data, pretrained_files=pretrained_files, include_full_pretrained_words=False)
    model = build_model(vocab,
                        embedding_dim=300,
                        pretrained_file=pretrained_files["tokens"])
    if cuda_device >= 0:
        model = model.cuda(cuda_device)
    train_data.index_with(vocab)
    train_loader = build_data_loaders(train_data=train_data, batch_size=batch_size)

    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(
            model=model,
            serialization_dir=serialization_dir,
            train_loader=train_loader,
            num_epochs=5,
            cuda_device=cuda_device
        )
        print("Starting training")
        trainer.train()
        print("Finished training")

    # outputs = model.forward_on_instances(instances)
    # print(outputs)
    return model, dataset_reader


if __name__ == '__main__':
    model, dataset_reader = run_training_loop()
    vocab = model.vocab
    print("vocab size: ", vocab.get_vocab_size("tokens"))
    train_data = dataset_reader.read('/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.txt')
    train_data.index_with(vocab)
    data_loader = DataLoader(train_data, batch_size=8)

    results = evaluate(model, data_loader)
    print(results)

    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict("我叫天宏。 [SEP] 我很喜欢自然语言处理。")
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
           for label_id, prob in enumerate(output['probs'])])