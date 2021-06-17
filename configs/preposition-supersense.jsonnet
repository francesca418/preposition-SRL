{
    "dataset_reader": {
        "type": "preposition_supersense",
        "bert_model_name": "bert-base-uncased",
    },
 
    "iterator": {
        "type": "bucket",
        "batch_size": 100,
        "sorting_keys": [["tokens", "num_tokens"]]
    },
 
    "train_data_path": "data/updated-simplified-train.txt",
    "validation_data_path": "data/updated-simplified-dev.txt",
    "test_data_path": "data/updated-simplified-test.txt",
 "evaluate_on_test" : true,
    "model": {
        "type": "preposition_supersense_bert",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-uncased",
    },
 
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.001,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 250,
            "num_steps_per_epoch": 8829,
        },

        "num_serialized_models_to_keep": 1,
        "num_epochs": 250,
        "validation_metric": "+combined_accuracy_score",
        "should_log_learning_rate": true,
	"cuda_device": [0, 1, 2, 3]
    }
}