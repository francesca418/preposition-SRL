# preposition-SRL
Using this repository, you will be able to perform BERT-based preposition SRL, training on the Streusle 4.0 dataset, which has been both manually annotated for arguments, as well as annotated using a dependency parser.

## Relevant Papers
TODO -- add relevant papers

## Set Up Environment
Create the environment and install necessary packages.
```
python3 -m venv prep_srl_venv
cd prep_srl_venv
source bin/activate
cd ..
pip install allennlp==0.9.0
pip install allennlp-models==0.9.0
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone <this repository's link>
cd preposition-SRL
```

## Investigate the STREUSLE 4.0 Data
The pre-processed, annotated STREUSLE data is included in this repository, since it is so small. You can also access the original STREUSLE data at [this link](https://github.com/nert-nlp/streusle).

## To Train the SRL Model
Set up paths and files referenced in ```preposition-srl.jsonnet```. Set the ```$SRL_TRAIN_DATA_PATH```, ```$SRL_VALIDATION_DATA_PATH```, and ```$SRL_TEST_DATA_PATH```.
Run ```. ./set_variables.sh``` Feel free to update any of the paths to your needs.

Navigate back to the outer ```preposition-srl``` directory.

Train the model:
```
allennlp train configs/preposition-srl.jsonnet -s prep-srl-test -f --include-package prep_srl
```

## To Evaluate the SRL Model
```
allennlp evaluate prep-srl-test/model.tar.gz $SRL_TEST_DATA_PATH --output-file prep-srl-test/evaluation.txt --include-package prep_srl
```

## To Predict SRL -- TODO: UPDATE
Create input text file with JSON formatting: ```{"sentence": "This is a sentence."}``` for each sentence you would like predicted.
```
allennlp predict prep-srl-test/model.tar.gz input.txt --output-file predicted_output.txt --predictor "preposition-srl" --include-package prep_srl
```

## Supersense Models
Train and evaluate preposition supersense models by following the above instructions, substituting in the appropriate data and models. (These models do not currently produce results that beat the best published results.)

## Joint SRL & Supersense Models
Train and evaluate preposition SRL and supersense models by following the above instructions, substituting in the appropriate data and models. (These models do not currently produce strong results.)
