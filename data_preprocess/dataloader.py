from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch


def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组(seq, label)，第一个元素：句子序列数据， 第二个元素：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data为一个batch的数据组成的列表，batch_data中某一元素的形式如 (list(wordids), seq_len, label)
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数默认要求序列按照长度降序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [torch.LongTensor(xi[0]) for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
    return padded_sent_seq, data_length, torch.tensor(label, dtype=torch.int64)


class classifier_dataset(Dataset):
    '''
    :param text (list): 已经tokenize好的N条句子，包含N个list
    :param label (list): 标签，包含N个int表示类别，假设为C分类，则元素为0 ~ C-1
    加载用于文本分类的数据，默认有text和label两个array，text已经通过了tokenizer变为了wordid
    '''
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return (self.text[item], self.label[item])

# if __name__ == '__main__':
#     train_x = [[1, 1, 1, 1, 1, 1, 1],
#                [2, 2, 2, 2, 2, 2],
#                [3, 3, 3, 3, 3],
#                [4, 4, 4, 4],
#                [5, 5, 5],
#                [6, 6],
#                [7]]
#     train_y = [1, 2, 1, 2, 3, 0, 1]
#     train_data = classifier_dataset(train_x, train_y)
#     # 将dataset封装为data_loader
#     data_loader = DataLoader(dataset=train_data,
#                              batch_size=3,
#                              collate_fn=collate_fn,
#                              shuffle=True)
#     from models.LSTM import LSTMModel
#     model = LSTMModel(input_size=1, hidden_size=10)
#     for batch in data_loader:
#         batch_x, batch_x_len, batch_y = batch
#         batch_x_pack = pack_padded_sequence(batch_x.unsqueeze(-1).float(), batch_x_len, batch_first=True)
#         outputs, hn, cn = model(batch_x_pack)
#         outputs_pad, outputs_len = pad_packed_sequence(outputs, batch_first=True)

