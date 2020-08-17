{
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
  "train_data_path": "/Users/tianhongzxy/Downloads/contradictory-my-dear-watson/train-en.txt",
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
        "type": "bag_of_embeddings",
        "embedding_dim": 10,
        // "input_size": 10,
        // "hidden_size": 10
    }
  },
  "data_loader": {
    "batch_size": 8,
    "shuffle": true
  },

  "trainer": {
    "num_epochs": 5,
    "optimizer": {
      "type": "adam",
    }
  }
}