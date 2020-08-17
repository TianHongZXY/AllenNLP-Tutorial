import os
import jieba
import jieba.posseg as poss
from typing import List, Tuple, Union
from allennlp.data.tokenizers import Tokenizer, Token


@Tokenizer.register('jieba')
class JiebaTokenzer(Tokenizer):
    """
    A ``Tokenizer`` that uses JIEBA's tokenizer. To Split Chinese sentences.
    :user_dict: a txt file, one word in a line.
    """
    def __init__(self,
                 pos_tags: bool = False,
                 user_dict: str = None
                 ) -> None:
        self._pos_tags = pos_tags

        if user_dict and os.path.exists(user_dict):
            jieba.load_userdict(user_dict)

    def _sanitize(self, tokens: Union[List[str], List[Tuple[str, str]]]) -> List[Token]:
        """
        Converts tokens to allennlp Token.
        """
        sanitize_tokens = []
        if self._pos_tags:
            for text, pos in tokens:
                token = Token(text=text, pos_=pos)
                sanitize_tokens.append(token)
        else:
            for text in tokens:
                token = Token(text)
                sanitize_tokens.append(token)
        return sanitize_tokens

    def tokenize(self, text: str) -> List[Token]:
        if self._pos_tags:
            return self._sanitize(poss.cut(text))
        else:
            return self._sanitize(jieba.cut(text))


# if __name__ == '__main__':
#     text = "我叫天宏，我很喜欢自然语言处理。"
#     tokenizer = JiebaTokenzer(pos_tags=True)
#     tokens = tokenizer.tokenize(text)
#     print('/'.join([x.text for x in tokens]))
#     print('/'.join([x.pos_ for x in tokens]))
