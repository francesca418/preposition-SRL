# supersense-data

The data named train.txt, development.txt, test.txt are used for training models to predict preposition supersense, based on the labeling schema and training splits defined in the STREUSLE dataset. This is the same data that is contained in the ../srl-data/ directory.

The data named updated-train.txt, updated-dev.txt, updated-test.txt are used for training models to predict preposition supersense, based on the labeling schema defined in the STREUSLE dataset. These data have been slightly restructured so that supersense labels that only occur in the test or dev sets are moved into the train set.

The data named simplified-train.txt, simplified-dev.txt, simplified-test.txt are contain slightly simplified versions of the supersense labels (some labels are changed to a label higher in the supersense hierarchy in order to reduce the number of labels predicted in the hopes of increasing model performance).

The data named updated-simplified-*.txt are a combination of the updated data and the simplified data.