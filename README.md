# Legal language compression

*Repository under construction*

Full readme to follow after academic submission.

# Code and data list

### data/

#### data/raw_data

**File** | **Description**
---|---
data_split.ipynb | Script for splitting the raw (legislative) data into training, validation and test sets
leg_targetted_train_data.pickle | 'Targetted' sub-set of training data
leg_targetted_val_data.pickle | 'Targetted' sub-set of validation data
leg_test_data.pickle | Test data
leg_train_data.pickle | Training data
leg_val_data.pickle | Validation data
legislative_compressions_tsv | Raw corpus

#### data/BERT_parsed_data

**File** | **Description**
---|---
BERT_data_prep.ipynb | Script for preparing data for BERT input
leg_targetted_train_data_128_TFR | BERT input, 'targetted' sub-set of legislative training data
leg_targetted_val_data_128_TFR | BERT input, 'targetted' sub-set of legislative validation data
leg_test_data_128_TFR | BERT input, legislative test data
leg_train_data_128_TFR | BERT input, legislative training data
leg_val_data_128_TFR | BERT input, legislative validation data

#### data/LSTM_parsed_data

**File** | **Description**
---|---
LSTM_data_prep.ipynb | Script for preparing data for LSTM input
leg_embeddings* | Vocabulary and associated word2vec embeddings for the legislative input text
leg_test* | Legislative test data: labels, original text, spaCy parsed text and vocabulary
leg_train* | Legislative training data: labels, original text, spaCy parsed text and vocabulary
leg_train_targetted* | 'Targetted' sub-set of legislative training data: labels, original text, spaCy parsed text and vocabulary
leg_val* | Legislative validation data: labels, original text, spaCy parsed text and vocabulary
leg_val_targetted* | 'Targetted' sub-set of legislative validation data: labels, original text, spaCy parsed text and vocabulary
total_vocab | Total legislative dataset vocabulary

### models/

#### models/LSTM_models

The LSTM models are quite large (each in excess of 100MB) and instead of hosting them on GitHub, they are hosted in a publically accessible Google Storage Bucket:

- LSTM_base (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/LSTM_base>)
- LSTM_leg_only (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/LSTM_leg_only>)
- LSTM_transfer (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/LSTM_transfer>)
- LSTM_ULMFiT (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/LSTM_ulmfit>)

**File** | **Description**
---|---
model_test_script.ipynb | Script for testing the LSTM models
model_train_script.ipynb | Script for training the LSTM models
model_ulmfit_script.ipynb | Script imlpementing transfer learning using the ULMFiT procedures

#### models/Rules_models

**File** | **Description**
---|---
BERT_prep.py | A helper module for preparing BERT inputs as part of the BERT_rules_ensemble model
rule_transformations.py | The rule transformations implemented in the rules based model
rules_BERT_script.ipynb | A helper script for preparing BERT inputs as part of the BERT_rules_ensemble model
rules_only_script.ipynb | Script for running the rules based model

#### models/BERT_models

The BERT models are quite large (each in excess of 1GB) and instead of hosting them on GitHub, they are hosted in a publically accessible Google Storage Bucket:

- BERT_base (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/bert_base>)
- BERT_transfer (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/bert_transfer>)
- BERT_leg_only (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/bert_leg_only>)
- BERT_rules (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/bert_rules>)
- BERT_rules_leg_only (<https://console.cloud.google.com/storage/browser/rob-bert-legal-compression-public/public/bert_rules_leg_only>)

**File** | **Description**
---|---
model_run_script.ipynb | Script for training and testing the BERT models (designed to be run on Google Colaboratory)
beam_search.py | Beam search module for BERT models
model.py | Main decoder transformer model

### evaluation/

**File** | **Description**
---|---
BERT_reconstruct.py | A module to reconstruct the outputs from the BERT models
BLEU.ipynb | Script for calculating the BLEU evaluation metric
LSTM_reconstruct.py | A module to reconstruct the outputs from the LSTM models
PrecisionRecall.ipynb | Script for calculating precision, recall, F1, compression ratio and string-for-string match
ROUGE.ipynb | Script for calculating the ROUGE evaluation metric
SARI.ipynb | Script for calculating the SARI evaluation metric
SARI.py | Underlying module for SARI calculation

#### evaluation/results

**File** | **Description**
---|---
BERT_base.pickle | Test results for the BERT base model
BERT_leg_only.pickle | Test results for the BERT leg only model
BERT_rules.pickle | Test results for the BERT rules ensemble model (with pre-training)
BERT_rules_leg_only.pickle | Test results for the BERT rules ensemble model (without pre-training)
BERT_transfer.pickle | Test results for the BERT transfer model
LSTM_base.pickle | Test results for the LSTM base model
LSTM_leg_only.pickle | Test results for the LSTM leg only model
LSTM_transfer.pickle | Test results for the LSTM transfer model
LSTM_ulmfit.pickle |Test results for the LSTM ULMFiT model
Rules.pickle | Test results for the rules based model

#### evaluation/extrinsic

**File** | **Description**
---|---
compressed_legislation.docx | Compressed legislation generated by 'BERT_rules_leg_only model' for extrinsic evaluation task
QA_responses.docx | The responses to the comprehension questions generated by the 'cape' QA model for both compressed and uncompressed legislation
uncompressed_legislation.docx | Uncompressed legislation for extrinsic evaluation task

The original surveys can be accessed here:

- <https://docs.google.com/forms/d/e/1FAIpQLSceEYbnpHR3K8y1PVMJ3dxlTwvOsGIE2hQN0hchbr-QC34xLw/viewform?usp=sf_link>
- <https://docs.google.com/forms/d/e/1FAIpQLSfZG7DKNoa1-A_ZNo6g6MIiHOiucf1nurWwAy2DJs_UQe5qwg/viewform?usp=sf_link>
