import torch
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.nn import util as nn_util

import warnings
warnings.filterwarnings("ignore")

# Splits text into words (instead of wordpieces or characters).
tokenizer = WhitespaceTokenizer()

# Represents each token with both an id from a vocabulary and a sequence of
# characters.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab')
}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ['This', 'is', 'some', 'text', '.'],
    namespace='token_vocab')
vocab.add_tokens_to_namespace(
    ['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
    namespace='character_vocab')

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# The setup here is the same as what we saw above.
text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
# Note now that we have two entries in this output dictionary,
# one for each indexer that we specified.
print("Combined tensor dictionary:", tensor_dict)

# Now we split text into words with part-of-speech tags, using Spacy's POS tagger.
# This will result in the `tag_` variable being set on each `Token` object, which
# we will read in the indexer.
tokenizer = SpacyTokenizer(pos_tags=True)
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

# Represents each token with (1) an id from a vocabulary, (2) a sequence of
# characters, and (3) part of speech tag ids.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
    'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                     feature_name='tag_'),
}

tokens = tokenizer.tokenize(text)
print("Spacy tokens:", tokens)
print("POS tags:", [token.tag_ for token in tokens])

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("Tensor dict with POS tags:", tensor_dict)





# Embedder

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# Note that we added the batch dimension at the front.  You choose the 'indexer1'
# name when you configure your data processing code.
token_tensor = {'indexer1': {'tokens': torch.LongTensor([[1, 3, 2, 9, 4, 3]])}}

# You would typically get the number of embeddings here from the vocabulary;
# if you use `allennlp train`, there is a separate process for instantiating the
# Embedding object using the vocabulary that you don't need to worry about for
# now.
embedding = Embedding(num_embeddings=10, embedding_dim=3)

# This 'indexer1' key must match the 'indexer1' key in the `token_tensor` above.
# We use these names to align the TokenIndexers used in the data code with the
# TokenEmbedders that do the work on the model side.
embedder = BasicTextFieldEmbedder(token_embedders={'indexer1': embedding})

embedded_tokens = embedder(token_tensor)
print("Using the TextFieldEmbedder:", embedded_tokens)

# As we've said a few times, what's going on inside is that we match keys between
# the token tensor and the token embedders, then pass the inner dictionary to the
# token embedder.  The above lines perform the following logic:
embedded_tokens = embedding(**token_tensor['indexer1'])
print("Using the Embedding directly:", embedded_tokens)

# This is what gets created by TextField.as_tensor with a TokenCharactersIndexer
# Note that we added the batch dimension at the front. Don't worry too much
# about the magic 'token_characters' key - that is hard-coded to be produced
# by the TokenCharactersIndexer, and accepted by TokenCharactersEncoder;
# you don't have to produce those yourself in normal settings, it's done for you.
token_tensor = {'indexer2': {'token_characters': torch.tensor(
    [[[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]]])}}

character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# Again here, the 'indexer2' key is arbitrary. It just has to match whatever key
# you gave to the corresponding TokenIndexer in your data code, which ends up
# as the top-level key in the token_tensor dictionary.
embedder = BasicTextFieldEmbedder(token_embedders={'indexer2': token_encoder})

embedded_tokens = embedder(token_tensor)
print("With a character CNN:", embedded_tokens)





# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer
# and a TokenCharactersIndexer; see the code snippet above. This time we're using
# more intuitive names for the indexers and embedders.
token_tensor = {
    'tokens': {'tokens': torch.LongTensor([[2, 4, 3, 5]])},
    'token_characters': {'token_characters': torch.LongTensor(
        [[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]])}
}

# This is for embedding each token.
embedding = Embedding(num_embeddings=6, embedding_dim=3)

# This is for encoding the characters in each token.
character_embedding = Embedding(num_embeddings=6, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# 自动concat到一起，变成3 + 4 = 7维
embedder = BasicTextFieldEmbedder(
    token_embedders={'tokens': embedding, 'token_characters': token_encoder})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer,
# a TokenCharactersIndexer, and another SingleIdTokenIndexer for PoS tags;
# see the code above.
token_tensor = {
    'tokens': {'tokens': torch.LongTensor([[2, 4, 3, 5]])},
    'token_characters': {'token_characters': torch.LongTensor(
        [[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]])},
    'pos_tag_tokens': {'tokens': torch.tensor([[2, 5, 3, 4]])}
}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ['This', 'is', 'some', 'text', '.'],
    namespace='token_vocab')
vocab.add_tokens_to_namespace(
    ['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
    namespace='character_vocab')
vocab.add_tokens_to_namespace(
    ['DT', 'VBZ', 'NN', '.'],
    namespace='pos_tag_vocab')

# Notice below how the 'vocab_namespace' parameter matches the name used above.
# We're showing here how the code works when we're constructing the Embedding from
# a configuration file, where the vocabulary object gets passed in behind the
# scenes (but the vocab_namespace parameter must be set in the config). If you are
# using a `build_model` method (see the quick start chapter) or instantiating the
# Embedding yourself directly, you can just grab the vocab size yourself and pass
# in num_embeddings, as we do above.

# This is for embedding each token.给了vocab和namespace就不用给num_embeddings了
embedding = Embedding(embedding_dim=3, vocab_namespace='token_vocab', vocab=vocab)

# This is for encoding the characters in each token.
character_embedding = Embedding(embedding_dim=4,
                                vocab_namespace='character_vocab',
                                vocab=vocab)
cnn_encoder = CnnEncoder(embedding_dim=4, num_filters=5, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# This is for embedding the part of speech tag of each token.
pos_tag_embedding = Embedding(embedding_dim=6,
                              vocab_namespace='pos_tag_vocab',
                              vocab=vocab)

# Notice how these keys match the keys in the token_tensor dictionary above;
# these are the keys that you give to your TokenIndexers when constructing
# your TextFields in the DatasetReader.
embedder = BasicTextFieldEmbedder(
    token_embedders={'tokens': embedding,
                     'token_characters': token_encoder,
                     'pos_tag_tokens': pos_tag_embedding})
# 自动concat为3 + 5 + 6 = 14维
embedded_tokens = embedder(token_tensor)
print(embedded_tokens)





# Splits text into words (instead of wordpieces or characters).  For ELMo, you can
# just use any word-level tokenizer that you like, though for best results you
# should use the same tokenizer that was used with ELMo, which is an older version
# of spacy.  We're using a whitespace tokenizer here for ease of demonstration
# with binder.
tokenizer = WhitespaceTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer = ELMoTokenCharactersIndexer().tokens_to_indices

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to add
# anything, but we do need to construct the vocab object so we can use it below.
# (And if you have any labels in your data that need indexing, you'll still need
# this.)
vocab = Vocabulary()

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)

text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("ELMo tensors:", tensor_dict)

# Any transformer model name that huggingface's transformers library supports will
# work here.  Under the hood, we're grabbing pieces from huggingface for this
# part.
transformer_model = 'bert-base-cased'

# To do modeling with BERT correctly, we can't use just any tokenizer; we need to
# use BERT's tokenizer.
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)

# Represents each wordpiece with an id from BERT's vocabulary.
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)

text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("BERT tokens:", tokens)

text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("BERT tensors:", tensor_dict)

# Now we'll do an example with paired text, to show the right way to handle [SEP]
# tokens in AllenNLP.  We have built-in ways of handling this for two text pieces.
# If you have more than two text pieces, you'll have to manually add the special
# tokens.  The way we're doing this requires that you use a
# PretrainedTransformerTokenizer, not the abstract Tokenizer class.

# Splits text into wordpieces, but without adding special tokens.
tokenizer = PretrainedTransformerTokenizer(
    model_name=transformer_model,
    add_special_tokens=False,
)

context_text = "This context is frandibulous."
question_text = "What is the context like?"
context_tokens = tokenizer.tokenize(context_text)
question_tokens = tokenizer.tokenize(question_text)
print("Context tokens:", context_tokens)
print("Question tokens:", question_tokens)

combined_tokens = tokenizer.add_special_tokens(context_tokens, question_tokens)
print("Combined tokens:", combined_tokens)

text_field = TextField(combined_tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("Combined BERT tensors:", tensor_dict)





# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer = WhitespaceTokenizer()
token_indexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()
text = "This is some text."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)
text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("ELMo tensors:", token_tensor)

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json'
elmo_weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5'
elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("ELMo embedded tokens:", embedded_tokens)


# Again, it's easier to just run the data code to get the right output.

# We're using the smallest transformer model we can here, so that it runs on
# binder.
transformer_model = 'google/reformer-crime-and-punishment'
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)
text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("Transformer tokens:", tokens)
text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("Transformer tensors:", token_tensor)

embedding = PretrainedTransformerEmbedder(pretrained_model=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Transformer embedded tokens:", embedded_tokens)