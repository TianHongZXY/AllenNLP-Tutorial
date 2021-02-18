from typing import Dict, Iterable

from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
import glob
import os
import io


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        # with open(file_path, "r") as lines:
        #     for line in lines:
        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(file_path, label, '*.txt')):
                with io.open(fname, 'r', encoding="utf-8") as f:
                    text = f.readline()
                    yield self.text_to_instance(text, label)


if __name__ == '__main__':
    dataset_reader = ClassificationTsvReader()
    train_data = dataset_reader.read('.data/imdb/aclImdb/train')
    test_data = dataset_reader.read('.data/imdb/aclImdb/test')

    for i, ins in enumerate(train_data):
        print(i, ins)
        if i == 10:
            break

