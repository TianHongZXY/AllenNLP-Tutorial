import csv
from typing import Dict, Iterable, List

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp_models.generation import Seq2SeqDatasetReader as Seq2SeqDatasetReaderV2


class Seq2SeqDatasetReaderV1(DatasetReader):
    def __init__(self,
                source_tokenizer: Tokenizer = None,
                target_tokenizer: Tokenizer = None,
                source_token_indexers: Dict[str, TokenIndexer] = None,
                target_token_indexers: Dict[str, TokenIndexer] = None,
                **kwargs,
                ) -> None:
       super().__init__(**kwargs)
       self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
       self._target_tokenizer = target_tokenizer or self._source_tokenizer
       self._source_token_indexer = (source_token_indexers
                                     or {"tokens": SingleIdTokenIndexer()})
       self._target_token_indexers = (target_token_indexers
                                      or self._source_token_indexer)
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as data_file:
            for line_num, row in enumerate(csv.reader(data_file, delimiter='\t')):
                source_sequence, target_sequence = row
                yield self.text_to_instance(source_sequence, target_sequence)

    def text_to_instance(self, source_string: str, target_string: str = None
            ) -> Instance:
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        source_field = TextField(tokenized_source, self._source_token_indexer)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({"source_tokens": source_field})
        

if __name__ == "__main__":
    source_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="source_tokens"),
                             "character_tokens": TokenCharactersIndexer(namespace="source_char_tokens")}
    target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="target_tokens"),
                             "character_tokens": TokenCharactersIndexer(namespace="target_char_tokens")}
    dataset_reader_v1 = Seq2SeqDatasetReaderV1(source_token_indexers=source_token_indexers,
                                               target_token_indexers=target_token_indexers)
    dataset_reader_v2 = Seq2SeqDatasetReaderV2(source_token_indexers=source_token_indexers,
                                               target_token_indexers=target_token_indexers)
    instances = list(dataset_reader_v2.read(args.train_file))
    vocab = Vocabulary.from_instances(instances)
    print(vocab)
    print(vocab.get_namespaces())

    for i, instance in enumerate(instances):
        if i == 3:
            break
        print(instance)

