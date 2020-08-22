import torch
import json
import tempfile
import numpy as np
from copy import deepcopy
from itertools import islice
from typing import Iterable, Dict, List, Tuple, Optional
from allennlp.common import JsonDict, Params
from allennlp.common.file_utils import cached_path
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.initializers import InitializerApplicator, XavierUniformInitializer, ConstantInitializer, NormalInitializer
from allennlp.nn.regularizers import RegularizerApplicator, L1Regularizer, L2Regularizer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Instance, DatasetReader, TextFieldTensors, DataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp_tutorial.JiebaTokenizer.jiebatokenizer import JiebaTokenzer
from allennlp_tutorial.BertTokenizer.berttokenizer import BERTTokenizer
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data.vocabulary import Vocabulary, _read_pretrained_tokens
from allennlp.data.token_indexers import TokenIndexer, SpacyTokenIndexer, SingleIdTokenIndexer, \
    ELMoTokenCharactersIndexer, PretrainedTransformerIndexer
from allennlp.models import Model
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder, PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.training.util import evaluate
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.predictors import Predictor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None,
                       token_indexers: Dict[str, TokenIndexer] = None,
                       **kwargs):
        """
        :param tokenizer: used to split text into list of tokens
        :param token_indexers: it defines how to map tokens to integer
        """
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        text_field = TextField(tokens, self._token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str, sep: str = '\t') -> Iterable[Instance]:
        '''
        :param file_path: a local file path or a url
        :param sep: by default separate row with '\t' to read tsv, also you can pass ',' to read csv
        '''
        with open(cached_path(file_path), 'r') as lines:
            for line in islice(lines, 1, None):
                text, _, __, label = line.strip().split(sep)
                yield self.text_to_instance(text, label)

    # def _read(self, file_path: str) -> Iterator[Instance]:
    #     """使用pandas读取文件"""
    #     import pandas as pd
    #     df = pd.read_csv(file_path, sep='\t')
    #     # if config.testing: df = df.head(1000)
    #     for i, row in df.iterrows():
    #         yield self.text_to_instance(text=self._tokenizer.tokenize(row["text"]),
    #                                     label=row["label"])


CONFIG = """{
    "dataset_reader": {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
        "tokenizer": {
            "type": "spacy"
        }
    },
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "rnn",
            "hidden_size": 10,
            "input_size": 10
        }
    },
    "train_data_path": "/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.txt",
    "trainer": {
        "num_epochs": 5,
        "optimizer": {
            "type": "adam"
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    }
}"""


def run_config(config):
    params = Params(json.loads(config))
    params_copy = params.duplicate()

    if 'dataset_reader' in params:
        reader = DatasetReader.from_params(params.pop('dataset_reader'))
    else:
        raise RuntimeError('`dataset_reader` section is required')

    all_instances = []
    if 'train_data_path' in params:
        print('Reading the training data...')
        train_data = reader.read(params.pop('train_data_path'))
        all_instances.extend(train_data)
    else:
        raise RuntimeError('`train_data_path` section is required')

    validation_data = None
    if 'validation_data_path' in params:
        print('Reading the validation data...')
        validation_data = reader.read(params.pop('validation_data_path'))
        all_instances.extend(validation_data)

    print('Building the vocabulary...')
    vocab = Vocabulary.from_instances(all_instances)

    model = None
    iterator = None
    if 'model' not in params:
        # 'dataset' mode — just preview the (first 10) instances
        print('Showing the first 10 instances:')
        for inst in all_instances[:10]:
            print(inst)
    else:
        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        loader_params = deepcopy(params.pop("data_loader"))
        train_data_loader = DataLoader.from_params(dataset=train_data,
                                                   params=loader_params)
        dev_data_loader = DataLoader.from_params(dataset=validation_data,
                                                 params=loader_params)
        train_data.index_with(vocab)

        # set up a temporary, empty directory for serialization
        with tempfile.TemporaryDirectory() as serialization_dir:
            trainer = Trainer.from_params(
                model=model,
                serialization_dir=serialization_dir,
                data_loader=train_data_loader,
                validation_data_loader=dev_data_loader,
                params=params.pop('trainer'))
            trainer.train()

    return {
        'params': params_copy,
        'dataset_reader': reader,
        'vocab': vocab,
        'iterator': iterator,
        'model': model
    }

def save_model():
    # Save the model
    serialization_dir = 'model'
    config_file = os.path.join(serialization_dir, 'config.json')
    vocabulary_dir = os.path.join(serialization_dir, 'vocabulary')
    weights_file = os.path.join(serialization_dir, 'weights.th')

    os.makedirs(serialization_dir, exist_ok=True)
    params.to_file(config_file)
    vocab.save_to_files(vocabulary_dir)
    torch.save(model.state_dict(), weights_file)

    # Load the model
    loaded_params = Params.from_file(config_file)
    loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
    loaded_vocab = loaded_model.vocab  # Vocabulary is loaded in Model.load()

    # Make sure the predictions are the same
    loaded_preds = make_predictions(loaded_model, dataset_reader)
    assert original_preds == loaded_preds
    print('predictions matched')

    # Create an archive file
    archive_model(serialization_dir, weights='weights.th')

    # Unarchive from the file
    archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'))


def read_data(reader: DatasetReader) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    # training_data = reader.read("quick_start/data/movie_review/train.tsv")
    # validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
    return training_data, validation_data


@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = None,
                 **kwargs
                 ):
        super().__init__(vocab, **kwargs)
        self.embedder = embedder
        self.encoder = encoder
        self.vocab = vocab
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        if initializer:
            initializer(self)

    def forward(self, text: TextFieldTensors,
                      label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Note that the signature of forward() needs to match that of field names
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
            loss_reg = self.get_regularization_penalty()
            outputs['loss'] = loss_reg + torch.nn.functional.cross_entropy(logits, label)
        return outputs

    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Take the logits from the forward pass, and compute the label
        # IDs for maximum values
        probs = output_dict['probs'].cpu().data.numpy()
        predicted_id = np.argmax(probs, axis=-1)
        # Convert these IDs back to label strings using vocab
        output_dict['label'] = [
            self.vocab.get_token_from_index(x, namespace='labels')
            for x in predicted_id
        ]
        return output_dict

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


def make_predictions(model: Model, dataset_reader: DatasetReader) \
        -> List[Dict[str, float]]:
    """Make predictions using the given model and dataset reader."""
    vocab = model.vocab
    predictions = []
    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict('A good movie!')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    output = predictor.predict('This was a monstrous waste of time.')
    predictions.append({vocab.get_token_from_index(label_id, 'labels'): prob
                        for label_id, prob in enumerate(output['probs'])})
    return predictions


def build_vocab(instances: Iterable[Instance],
                pretrained_files: Optional[Dict[str, str]] = None,
                include_full_pretrained_words: bool = False
                ) -> Vocabulary:
    print("Building the vocabulary")
    vocab = Vocabulary.from_instances(instances, min_count={"tokens": 1})
    if pretrained_files and include_full_pretrained_words:
        pretrained_tokens = _read_pretrained_tokens(pretrained_files["tokens"])
        from collections import Counter
        c = Counter(pretrained_tokens)
        counter = {"tokens": dict(c)}
        vocab._extend(counter=counter)
    print("Vocab size: ", vocab.get_vocab_size("tokens"))
    return vocab


def build_model(
        vocab: Vocabulary,
        embedding_dim: int,
        pretrained_file: str = None,
        initializer: InitializerApplicator = None,
        regularizer: RegularizerApplicator = None
        ) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    word_vec = Embedding(embedding_dim=embedding_dim,
                          num_embeddings=vocab_size,
                          pretrained_file=pretrained_file,
                          vocab=vocab)
    embedding = BasicTextFieldEmbedder({"tokens": word_vec})

    # Use ELMo
    # options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    # weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    # elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    # embedding = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    # Use BERT
    # bert_embedder = PretrainedTransformerEmbedder(
    #     model_name='bert-base-uncased',
    #     max_length=512,
    #     train_parameters=False
    # )
    # embedding = BasicTextFieldEmbedder({"tokens": bert_embedder})

    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim)
    return SimpleClassifier(vocab, embedding, encoder, initializer, regularizer=regularizer)


def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset = None,
    batch_size: int = 8):
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    batch_sampler = BucketBatchSampler(train_data, batch_size=batch_size, sorting_keys=["text"], padding_noise=0)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler)
    if dev_data:
        dev_batch_sampler = BucketBatchSampler(dev_data, batch_size=batch_size, sorting_keys=["text"], padding_noise=0)
        dev_loader = DataLoader(dev_data, batch_sampler=dev_batch_sampler)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader = None,
    num_epochs: int = 1,
    cuda_device: int = -1,
    patience: int = None
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
        cuda_device=cuda_device,
        patience=patience
    )
    return trainer


def run_training_loop():
    tokenizer = BERTTokenizer(vocab_file='/Users/tianhongzxy/Downloads/BiSentESIM/BiSentESIM/My-pipeline/allennlp_tutorial/BertTokenizer/vocab.txt')
    # tokenizer = BERTTokenizer('bert-base-multilingual-cased') # same as above

    # Try to use ELMo
    # tokenindexer = ELMoTokenCharactersIndexer()
    # elmo_tokens = tokenindexer.tokens_to_indices([Token("happy")], None)
    # print(len(elmo_tokens["elmo_tokens"][0]), elmo_tokens)

    # Try to use BERT
    # tokenizer = PretrainedTransformerTokenizer(
    #     model_name="bert-base-multilingual-cased",
    #     add_special_tokens=True,
    #     max_length=512
    # )
    # token_indexer = PretrainedTransformerIndexer(
    #     model_name="bert-base-multilingual-cased",
    #     max_length=512,
    # )

    cached_directory = None # "cached_dir"
    dataset_reader = ClassificationTsvReader(tokenizer=tokenizer, cache_directory=cached_directory)
    print("Reading data")
    train_data = dataset_reader.read(file_path='/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.txt')
    pretrained_files = None # {"tokens": "/Users/tianhongzxy/Downloads/BiSentESIM/BiSentESIM/embedding/glove.6B.300d.txt"}
    cuda_device = -1
    batch_size = 8
    vocab = build_vocab(train_data, pretrained_files=pretrained_files, include_full_pretrained_words=False)
    init_uniform = XavierUniformInitializer()
    # init_uniform(model.embedder.token_embedder_tokens.weight)
    init_const = ConstantInitializer(val=0)
    # init_const(model.classifier.bias)
    init_normal = NormalInitializer(mean=0., std=1.)
    # init_normal(model.classifier.weight)
    applicator = InitializerApplicator(
        regexes=[
            ('embedder.*', init_uniform),
            ('classifier.*weight', init_normal),
            ('classifier.*bias', init_const)
        ]
    )
    regularizer = RegularizerApplicator(
        regexes=[
            ('embedder.*', L2Regularizer(alpha=1e-3)),
            ('classifier.*weight', L2Regularizer(alpha=1e-3)),
            ('classifier.*bias', L1Regularizer(alpha=1e-2))
        ]
    )
    model = build_model(vocab,
                        embedding_dim=10,
                        pretrained_file=None, # pretrained_files["tokens"]
                        initializer=applicator,
                        regularizer=regularizer
                        )
    if cuda_device >= 0:
        model = model.cuda(cuda_device)

    # split train data into train & dev data
    from allennlp.data.dataset_readers import AllennlpDataset
    print('origin train data size: ', len(train_data))
    train_data, dev_data = train_test_split(train_data, test_size=0.2, random_state=20020206)
    assert type(train_data[0]) == type(dev_data[0]) == Instance
    train_data, dev_data = AllennlpDataset(train_data), AllennlpDataset(dev_data)
    print('train data size: ', len(train_data), 'dev data size', len(dev_data))
    assert type(train_data) == type(dev_data) == AllennlpDataset
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_loader, dev_loader = build_data_loaders(train_data=train_data,
                                                  dev_data=dev_data,
                                                  batch_size=batch_size)

    with tempfile.TemporaryDirectory() as serialization_dir:
        # serialization_dir = 'temp_dir/'
        trainer = build_trainer(
            model=model,
            serialization_dir=serialization_dir,
            train_loader=train_loader,
            dev_loader=dev_loader,
            num_epochs=5,
            cuda_device=cuda_device,
            patience=5
        )
        print("Starting training")
        trainer.train()
        print("Finished training")

    # outputs = model.forward_on_instances(instances)
    # print(outputs)
    return model, dataset_reader


if __name__ == '__main__':
    # components = run_config(CONFIG)
    # params = components['params']
    # dataset_reader = components['dataset_reader']
    # vocab = components['vocab']
    # model = components['model']

    model, dataset_reader = run_training_loop()
    vocab = model.vocab

    # Here's how to save the model.
    os.makedirs("temp_dir", exist_ok=True)
    with open("temp_dir/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("temp_dir/vocabulary")

    # print("vocab size: ", vocab.get_vocab_size("tokens"))
    # train_data = dataset_reader.read('/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train.txt')
    # train_data.index_with(vocab)
    # data_loader = DataLoader(train_data, batch_size=8)

    # results = evaluate(model, data_loader)
    # print(results)

    predictor = SentenceClassifierPredictor(model, dataset_reader)
    output = predictor.predict("我叫天宏。 [SEP] 我很喜欢自然语言处理。")
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
           for label_id, prob in enumerate(output['probs'])])