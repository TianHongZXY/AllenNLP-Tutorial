import torch
import torch.nn as nn
from models.LSTM import LSTMModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class nlimodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(nlimodel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        self.lstm = LSTMModel(input_size=embedding_dim,
                              hidden_size=hidden_size)
        # 教训，不要在输出标签的分类层后加relu！！！
        self.mlp = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=output_size))

    def forward(self, tinputs, tinputs_len):
        word_embedding = self.embedding(tinputs)
        # batch_x_pack = pack_padded_sequence(word_embedding, tinputs_len, batch_first=True)

        # hidden, h_n, c_n = self.lstm(word_embedding)
        # outputs = self.mlp(h_n.squeeze())
        outputs = torch.sum(word_embedding, dim=1)
        outputs = self.mlp(outputs)
        return outputs

