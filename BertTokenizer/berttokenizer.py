from typing import List
from allennlp.data.tokenizers import Tokenizer, Token
from transformers import BertTokenizer


class BERTTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that uses BERT's tokenizer.
    :param vocab_file: a path to a vocab file(one word per line) or
                       a model name like 'bert-base-uncased'
    """
    def __init__(self,
                 vocab_file: str,
                 ):
        self._tokenizer = BertTokenizer.from_pretrained(vocab_file)

    def tokenize(self, text: str) -> List[Token]:
        text = self._tokenizer.tokenize(text)
        return [Token(t) for t in text]


# if __name__ == '__main__':
#     tokenizer = BERTTokenizer('vocab.txt')
#     text = "我叫天宏，我很喜欢自然语言处理。"
#     text = tokenizer.tokenize(text=text)
#     print(text)
