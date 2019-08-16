# A module to reconstruct LSTM outputs

import re

def LSTM_reconstruct(prediction, target):

    # Remove trailing </S> and stringify original

    if len(prediction) != 0:
        if prediction[-1] == '</S>':
            prediction = prediction[:-1]

    if len(target) != 0:
        if target[-1] == '</S>':
            target = target[:-1]

    pred_reconstruction = " ".join(prediction)  # join words together
    pred_reconstruction = re.sub(' (?=[\.,):;\'\”£$€\-—])', '', pred_reconstruction)  # deal with right spaced punctuation
    pred_reconstruction = re.sub('(?<=[\-(\“]) ', '',
                            pred_reconstruction)  # deal with left spaced punctuation. exception needed for start of sequence
    pred_reconstruction = re.sub('\' s', '\'s', pred_reconstruction)  # deal with posessives

    pred_target = " ".join(target)  # join words together
    pred_target = re.sub(' (?=[\.,):;\'\”£$€\-—])', '', pred_target)  # deal with right spaced punctuation
    pred_target = re.sub('(?<=[\-(\“]) ', '',
                            pred_target)  # deal with left spaced punctuation. exception needed for start of sequence
    pred_target = re.sub('\' s', '\'s', pred_target)  # deal with posessives

    return pred_reconstruction, pred_target

def remove_padding(true_labels, predicted_labels):
    non_pad_labels = []
    non_pad_preds = []
    for i in range(len(true_labels)):
        eos = true_labels[i].argmax(axis=0)
        non_pad_labels.append(true_labels[i][0:eos])
        non_pad_preds.append(predicted_labels[i][0:eos])
    return non_pad_labels, non_pad_preds