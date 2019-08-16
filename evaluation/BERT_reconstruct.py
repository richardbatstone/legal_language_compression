# A module to reconstruct BERT outputs

import re
import pickle
import tokenization # Can be downloaded from https://github.com/google-research/bert
import numpy as np

# Broadly, to reconstruct the output, we must first recover BERT processed text (original, for label application, and target, for 
# comparisson). We do this by tokenizing the inputs and applying the labels again.
# 

def get_test_data():
    data_path = "" # Path to raw test data
    with open(data_path + "leg_test_data.pickle", 'rb') as f:
        data = pickle.load(f)
        tokenizer = tokenization.FullTokenizer(vocab_file=data_path+"vocab.txt", do_lower_case=True)
        #
        # The vocab file is the vocab used by BERT for wordpiece to ID lookup, which can be downloaded as part of the BERT base model:
        # https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
        #
        original_text = []
        compressed_text = []
        token_original = []
        for i in data:
            original_text.append(data[i]['full_text'])
            token_original.append(tokenizer.tokenize(data[i]['full_text']))
            compressed_text.append(data[i]['compressed_text'])
    return original_text, token_original, compressed_text


def reconstruct_sent(token_sent, original_sent):

    # The main reconstructions function

    re_sent = " ".join(token_sent) # join words together
    re_sent = re.sub(' ##','',re_sent) # join word pieces together
    re_sent = recover_caps(re_sent, original_sent)
    re_sent = re.sub(' (?=[\.,):;\'\”£$€\-—])','', re_sent) # deal with right spaced punctuation
    re_sent = re.sub('(?<=[\-(\“]) ','', re_sent) # deal with left spaced punctuation. exception needed for start of sequence
    re_sent = re.sub('\' s', '\'s', re_sent) # deal with posessives

    return re_sent

def recover_caps(partial_recon_sent, original_sent):

    # BERT tokenizer converts tokens to lowercase. This function attempts to restore case.

    caps = {}
    for word in original_sent.split(" "):
        if re.match('[A-Z]|\W[A-Z]', word) != None:
            word = re.sub('\W', '', word)
            if word in caps:
                caps[word] = caps[word] + 1
            else:
                caps[word] = 1
    for word in caps:
        if len(re.findall(word.lower(), partial_recon_sent)) == caps[word]:
            partial_recon_sent = re.sub(word.lower(), word, partial_recon_sent)
    return partial_recon_sent

def label_normalisation(token_sentences, token_labels):

    # BERT tokenizer creates sub-word parts. This function ensures predicted output
    # does not remove part only of a word.

    seq_len = len(token_sentences) + 1
    if seq_len > 128:
        expand_labels = 2 * np.ones(seq_len - 1)
        for j in range(len(token_labels)):
            expand_labels[j] = token_labels[j]
        token_labels = np.copy(expand_labels)
    else:
        pass

    ranges = []
    new_range = []
    start_range = True
    for i in range(len(token_sentences)):
        try:
            if token_sentences[i][0:2]=="##" and start_range:
                new_range.append(i-1)
                start_range = False
            if token_sentences[i][0:2]=="##" and token_sentences[i+1][0:2]!="##":
                new_range.append(i)
                ranges.append(new_range)
                new_range = []
                start_range = True
        except IndexError:
            pass
    try:
        if ranges != []:
            for span in ranges:
                range_labels = token_labels[span[0]:span[1]]
                decided_label = int(round(np.mean(range_labels)))
                for j in span:
                    token_labels[j] = decided_label
    except IndexError:
        pass

    return token_labels

def BERT_reconstruct(test_outputs):

    original_text, token_original, compressed_text = get_test_data()

    # Reconstruction

    reconstructions = []

    for i in range(len(test_outputs['predicted_labels'])):

        # Label normalisation

        norm_token_labels = label_normalisation(token_original[i], test_outputs['predicted_labels'][i])

        # Apply labels and reconstruct

        predicted_sentence = []

        for j in range(len(token_original[i])):
            if norm_token_labels[j] == 2:
                predicted_sentence.append(token_original[i][j])
        re_sent = reconstruct_sent(predicted_sentence, original_text[i])
        reconstructions.append(re_sent)

    return reconstructions, compressed_text, original_text

def BERT_rules_reconstruct(test_outputs, rules_data):

    original_text, _, compressed_text = get_test_data()

    data_path = "" # Path to raw test data
    tokenizer = tokenization.FullTokenizer(vocab_file=data_path + "vocab.txt", do_lower_case=True)
    #
    # The vocab file is the vocab used by BERT for wordpiece to ID lookup, which can be downloaded as part of the BERT base model:
    # https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    #
    token_original = [tokenizer.tokenize(rules_data['predictions'][i]) for i in range(len(original_text))]

    # Reconstruction

    reconstructions = []

    for i in range(len(test_outputs['predicted_labels'])):

        # Label normalisation

        norm_token_labels = label_normalisation(token_original[i], test_outputs['predicted_labels'][i])

        # Apply labels and reconstruct

        predicted_sentence = []

        for j in range(len(token_original[i])):
            if norm_token_labels[j] == 2:
                predicted_sentence.append(token_original[i][j])
        re_sent = reconstruct_sent(predicted_sentence, original_text[i])
        reconstructions.append(re_sent)

    return reconstructions, compressed_text, original_text