import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp_models.generation import SimpleSeq2Seq, Bart
# from allennlp_models.generation.models.bart import BartEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder, Seq2SeqEncoder
from transformers import BartModel
from overrides.overrides import overrides
from allennlp_models.generation import ComposedSeq2Seq, SimpleSeq2Seq


class BartEncoder(Seq2SeqEncoder):
    def __init__(self, model_name, use_pretrained_embeddings=False):
        super().__init__()
        # or some flag that indicates the bart encoder in it's entirety could be used.
        if use_pretrained_embeddings:
            # will use the entire bart encoder including all embeddings
            bart = PretrainedTransformerEmbedder(model_name, sub_module="encoder")
        else:
            bart = BartModel.from_pretrained(model_name)
            self.bart_encoder.embed_tokens = lambda x: x
            self.bart_encoder.embed_positions = lambda x: torch.zeros(
                (x.shape[0], x.shape[1], self.hidden_dim), dtype=torch.float32
            )
        self.hidden_dim = bart.config.hidden_size
        self.bart_encoder = bart.transformer_model


def create_seq2seqmodel(vocab, src_embedders, tgt_embedders, hidden_dim=100, num_layers=1,
                        encoder=None, max_decoding_steps=20, beam_size=1, use_bleu=True, device=0):
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(src_embedders.get_output_dim(), hidden_dim, batch_first=True))
    model = SimpleSeq2Seq(vocab, src_embedders, encoder, max_decoding_steps, target_namespace="target_tokens",
                          target_embedding_dim=tgt_embedders.get_output_dim(), beam_size=beam_size,
                          use_bleu=use_bleu)
    # encoder = BartEncoder('facebook/bart-base', use_pretrained_embeddings=True)
    # encoder = PretrainedTransformerEmbedder(model_name='facebook/bart-base', sub_module="encoder")
    # model = Bart(model_name='facebook/bart-base', vocab=vocab, max_decoding_steps=max_decoding_steps,
    #              beam_size=beam_size, encoder=encoder)
    model.to(device)
    return model


