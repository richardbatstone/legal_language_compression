import spacy
from spacy.matcher import Matcher

def get_subtree(doc, sub_tree_indices=[]):

    # A helper function to return sub-tree indices

    for ind in sub_tree_indices:
        for child in doc[ind].children:
            sub_tree_indices.append(child.i)
    return sub_tree_indices

def rule_1(doc):

    # A rule for removing parentheses. E.g.
    #
    # Some text (and some brackets) followed by some more text (and some further brackets).
    #
    # ===>>>>
    #
    # Some text followed by some more text.

    open_para_indices = []
    close_para_indices = []
    removed_ranges = []
    for i in range(len(doc)):
        if doc[i].text == "(":
            open_para_indices.append(i)
        if  doc[i].text == ")":
            close_para_indices.append(i)
    pairs = []
    if len(open_para_indices) == len(close_para_indices):
        for i in range(len(open_para_indices)):
            try:
                if close_para_indices[i] < open_para_indices[i+1]:
                    pairs.append([open_para_indices[i], close_para_indices[i]])
            except IndexError:
                pairs.append([open_para_indices[i], close_para_indices[i]])
    for i in range(len(pairs)):
        if pairs[i][1] - pairs[i][0] > 2: # ignore cases where bracketing a number
            removed_ranges = removed_ranges + list(range(pairs[i][0],pairs[i][1]+1))

    return removed_ranges

def rule_2(doc):

    # A rule for removing sub-trees having "in respect" as their head.

    rels = []
    for i in range(len(doc)-1):
        if doc[i].text=="in" and doc[i+1].text=="respect":
            rels.append(i)
    removed_indices = get_subtree(doc, rels)
    return removed_indices

def rule_3(doc):

    # A rule for removing sub-trees having "for the purpose of" as their head.

    heads = []
    bounding_indices = []
    for i in range(len(doc)-2):
        if doc[i].text.lower()=="for" and doc[i+1].text=="the" and (doc[i+2].text=="purpose" or doc[i+2].text=="purposes"):
            heads.append(i+2)
            bounding_indices.append(i)
    removed_indices = get_subtree(doc, heads) + bounding_indices
    for i in removed_indices:
        try:
            if (i+1 not in removed_indices) and (doc[i+1].pos_ == "PUNCT"):
                removed_indices.append(i+1)
        except IndexError:
            pass
    return removed_indices

def rule_5(doc, matcher):

    # Wrapper rule for matcher_or1

    removed_indices = []
    matches = matcher(doc)
    for _, start, end in matches:
        removed_indices = removed_indices + list(range(start+1, end))
    return removed_indices

def rule_6(doc):

    # A rule for removing cross-reference type wording

    candidate_starts = []
    candidate_ends = []
    removed_indices = []
    for i in range(len(doc)-2):
        if (doc[i].text.lower() == "a") and (doc[i+1].text == "reference") and (doc[i+2].text == "in"):
            candidate_starts.append(i)
        if doc[i].text == "to":
            candidate_ends.append(i)
    if candidate_starts != []:
        removed_indices = list(range(candidate_starts[0], candidate_ends[0]+1))
    return removed_indices

def rule_7(doc, matcher):

    # Wrapper rule for matcher_phrases

    removed_indices = []
    matches = matcher(doc)
    for _, start, end in matches:
        removed_indices = removed_indices + list(range(start, end))
    return removed_indices

def rule_8(doc):

    # A rule for removing certain sub-clauses. E.g.
    #
    # An officer of a company, or a person acting on behalf of a company, commits an offence if he uses, or authorises
    # the use of, a seal purporting to be a seal of the company on which its name is not engraved as required by
    # subsection (2).
    #
    # ====>>>>>
    #
    # An officer of a company commits an offence if he uses, or authorises
    # the use of, a seal purporting to be a seal of the company on which its name is not engraved as required by
    # subsection (2).

    comma_idx = []
    removed_indices = []
    for i in range(len(doc)):
        if doc[i].text == ",":
            comma_idx.append(i)
    for j in range(len(comma_idx)):
        try:
            if (doc[comma_idx[j]+1].text == "or") and (doc[comma_idx[j]-1].lemma_ == doc[comma_idx[j+1]-1].lemma_):
                removed_indices = removed_indices + list(range(comma_idx[j], comma_idx[j+1]+1))
        except:
            pass
    return removed_indices

def rule_9(doc, matcher):

    # A wrapper for matcher_doublet

    removed_indices = []
    matches = matcher(doc)
    for _, start, end in matches:
        start_pos = doc[start+1].pos_
        end_pos = doc[end-1].pos_
        pos_match = start_pos == end_pos
        sim_score = doc[start+1].similarity(doc[end-1])
        if sim_score > 0.44 and pos_match and start_pos != "PROPN" and start_pos != "PUNCT":
            removed_indices = removed_indices + list(range(start+2, end))
    return removed_indices

############################################################################################################################
#
# Reconstruction functions: to return a "nice" output, the tokenizsed inputs must be reconstructed. Essentially, this involves 
# concatenating tokens with spaces in between. There are some other tricks to help with stray punctuation and other edge cases
#
############################################################################################################################

def get_target_indices(doc, removed_ranges):

    # Helper function for reconstructing outputs

    target_indices = []
    for i in range(len(doc)):
        if i not in removed_ranges:
            target_indices.append(i)
    return target_indices


def punct_check(target_words, doc):

    # Dealing with punctuation in output

    error_punct = []
    for i in range(len(target_words)):
        if i > 0:
            if (doc[target_words[i]].is_punct) and (doc[target_words[i]].text != doc[target_words[i]].text_with_ws):
                try:
                    if (doc[target_words[i + 1]].is_punct) and (
                            doc[target_words[i + 1]].text != doc[target_words[i + 1]].text_with_ws):
                        error_punct.append(i + 1)
                except IndexError:
                    pass

    count = 0
    for i in error_punct:
        j = i - count
        del target_words[j]
        count += 1

    end_puncts = [".", "!", "?"]
    if doc[target_words[-1]].text in end_puncts:
        if doc[target_words[-2]].is_punct and not doc[target_words[-2]].is_quote:
            del target_words[-2]
    if doc[target_words[-1]].text not in end_puncts:
        if doc[target_words[-1]].is_punct and not doc[target_words[-1]].is_quote:
            del target_words[-1]
    if doc[target_words[0]].is_punct and not (doc[target_words[0]].is_quote):
        del target_words[0]
    return target_words

def vowel_check(target_words, doc):

    # Dealing with stray vowels

    sub_words = {}
    vowels = ["a", "e", "i", "o", "u"]
    for i in range(len(target_words)-1):
        if doc[target_words[i]].text.lower() == "an" and doc[target_words[i+1]].text[0].lower() not in vowels:
            sub_words[target_words[i]] = "a"
        if doc[target_words[i]].text.lower() == "a" and doc[target_words[i+1]].text[0].lower() in vowels:
            sub_words[target_words[i]] = "an"
    return sub_words


def create_augmented_headline(target_words, doc):
    if len(target_words) == 0 or doc.text == '. . .':
        augmented_headline = ""
    elif len(target_words) < 3:
        augmented_headline = "".join(doc[i].text_with_ws for i in target_words)
    else:
        end_puncts = [".", "!", "?"]
        target_words = punct_check(target_words, doc)

        compressed_sentence = []
        sub_words = vowel_check(target_words, doc)

        for i in target_words[:-2]:
            if ((doc[i].text == doc[i].text_with_ws) and (i + 1 not in target_words)):
                if i in sub_words:
                    compressed_sentence.append(sub_words[i] + " ")
                else:
                    compressed_sentence.append(doc[i].text + " ")
            else:
                if i in sub_words:
                    compressed_sentence.append(sub_words[i] + " ")
                else:
                    compressed_sentence.append(doc[i].text_with_ws)

        if doc[target_words[-1]].text in end_puncts:
            compressed_sentence.append(doc[target_words[-2]].text)
            compressed_sentence.append(doc[target_words[-1]].text)
        else:
            if (doc[target_words[-2]].text == doc[target_words[-2]].text_with_ws) and (
                    target_words[-2] + 1 not in target_words):
                if target_words[-2] in sub_words:
                    compressed_sentence.append(sub_words[target_words[-2]] + " ")
                else:
                    compressed_sentence.append(doc[target_words[-2]].text + " ")
            else:
                if target_words[-2] in sub_words:
                    compressed_sentence.append(sub_words[target_words[-2]] + " ")
                else:
                    compressed_sentence.append(doc[target_words[-2]].text_with_ws)

            compressed_sentence.append(doc[target_words[-1]].text)

        augmented_headline = "".join(compressed_sentence)
    return augmented_headline

#####################################################################################################################################
#
# Main entry point function.
#
#####################################################################################################################################

def apply_rules(text_list, ids=None):

    # Load the language model and parse the texts

    nlp = spacy.load('en_core_web_md')
    doc_list = nlp.pipe(text_list)

    # Define matcher objects

    matcher_or1 = Matcher(nlp.vocab)
    pattern_or1 = [{}, {"ORTH": ','}, {}, {}, {"ORTH": 'or'}]
    matcher_or1.add("or1", None, pattern_or1)

    matcher_phrases = Matcher(nlp.vocab)
    pattern_p1 = [{"ORTH": ','}, {"ORTH": 'as'}, {"ORTH": 'the'}, {"ORTH": 'case'}, {"ORTH": 'may'},
                  {"ORTH": 'be'}, {"ORTH": ','}]
    pattern_p2 = [{"ORTH": ','}, {"ORTH": 'to'}, {"ORTH": 'any'}, {"ORTH": 'extent'}, {"ORTH": ','}]
    pattern_p3 = [{"ORTH": ','}, {"ORTH": 'so'}, {"ORTH": 'far'}, {"ORTH": 'as'}, {"ORTH": 'is'},
                  {"ORTH": 'reasonably'}, {"ORTH": 'practicable'}, {"ORTH": ','}]
    matcher_phrases.add("p1", None, pattern_p1)
    matcher_phrases.add("p1", None, pattern_p2)
    matcher_phrases.add("p1", None, pattern_p3)

    matcher_doublet = Matcher(nlp.vocab)
    pattern_d1 = [{"IS_PUNCT": False}, {"POS": "NUM", "OP": "!"},
                  {"ORTH": 'and'}, {"POS": "NUM", "OP": "!"}]
    matcher_doublet.add("d1", None, pattern_d1)

    # Apply rules and reconstruct output. The structure of the rules could obviously be much simplified and made much
    # more efficient.

    predictions = []
    predicted_labels = []

    for doc in doc_list:
        remove_1 = rule_1(doc)
        remove_2 = rule_2(doc)
        remove_3 = rule_3(doc)
        remove_5 = rule_5(doc, matcher_or1)
        remove_6 = rule_6(doc)
        remove_7 = rule_7(doc, matcher_phrases)
        remove_8 = rule_8(doc)
        remove_9 = rule_9(doc, matcher_doublet)
        removed_indices = remove_1 + remove_2 + remove_3 + remove_5 + remove_6 + remove_7 + remove_8 + remove_9

        target_indices = get_target_indices(doc, removed_indices)
        predicted_labels.append(target_indices)
        prediction = create_augmented_headline(target_indices, doc)
        predictions.append(prediction)

    return predictions, predicted_labels

def apply_rules_to_section_list(section_list):
    text_list = []
    id_list = []
    for i in section_list:
        text_list.append(i.text)
        id_list.append(i.ID)
    predictions, predicted_labels = apply_rules(text_list, ids=id_list)
    for i in range(len(section_list)):
        section_list[i].text = predictions[i]
    return section_list