# Legal language compression

*Repository under construction*

# Data

## Overview

data/clean_data contains a parallel corpus of legal rule compressions. The raw data is available as a tab seperated text file. Pre-split train, validation and test files are made available as pickle files (see the 'data_split.ipynb' script for details). 'Targetted' train and validation splits are also made available as pickle files.

Constructing a sentence compression model relies, at least to some extent, on a parallel corpus of pairs of uncompressed and compressed text. The largest such corpus is the 'news headline' corpus constructed by [Katja Filippova, Enrique Alfonseca and others](https://www.aclweb.org/anthology/papers/D/D15/D15-1042/), which has approximately 2 million examples. That corpus is not publically available but the researchers have made a sub-set of approximately 200,000 examples available. However, we do not expect a model trained only on this news headline corpus to perform well in the legal domain, because of the linguistic differences between legislative rules and news article language.

Instead, we require a parallel compression corpus for the legal domain. We are not aware of any such corpus and therefore construct one. The corpus consists of 1,000 compressed and uncompressed pairs of legislative provisions, which were manually produced.

The structure of the corpus is summarised in the table below. The corpus consists of a mix of syntactically complete sentences and sentence 'fragments'.

**Field name** | **Type** | **Description**
---|---|---
Reference | String | The legislative reference for the text
Original text | String | The uncompressed legislative text. Section numbers are removed except where the text is a concatenation of sub-sections, in which the sub-section numbers are retained. 
Compressed text | String | The compressed version of the legislative text.
Targetted | Boolean | Whether the example belongs to the targetted training / validation sub-set.
Fragment | Boolean | Whether the example is a fragment or a syntactically complete sentence.
Concatenation | Boolean | Whether the example is a concatenation of a list of provisions. 

Where fragments of a sentence are included, the syntactically complete sentence (being the concatenation of the sentence fragments making up that provision) is also included. An example of this is shown in the table below. There is necessarily some repetition between these syntactically complete sentences and the sentence fragments and the corpus is labelled so that either the complete sentences or the fragments can be isolated.

**Original text** | **Compressed text** | **Targetted** | **Fragment** | **Concatenation**
---|---|---|---|---
Case 4 is where— (a) R requests, agrees to receive or accepts a financial or other advantage, and (b) the request, agreement or acceptance itself constitutes the improper performance by R of a relevant function or activity. (*Section 2(3) Bribery Act 2010*) | Case 4 is where— (a) R accepts a financial or other advantage, and (b) the acceptance itself constitutes the improper performance by R of a relevant function. | No | No | Yes
R requests, agrees to receive or accepts a financial or other advantage, and (*Section 2(3)(a) BA 2010*) | R accepts a financial or other advantage, and | Yes | Yes | No
the request, agreement or acceptance itself constitutes the improper performance by R of a relevant function or activity. (*Section 2(3)(b) BA 2010*) | the acceptance itself constitutes the improper performance by R of a relevant function. | Yes | Yes | No

There are 526 sentence fragments and 474 syntactically complete sentences in the corpus. 173 of the syntactically complete sentences are concatenations of provisions. 742 of the examples are marked as 'targetted' examples for training and validation purposes (discussed further below).

The corpus is not balanced in the sense that it contains only minimal examples of provisions which are not capable of compression. (The only examples of this type are where individual fragments of a compressed provision are not capable of being compressed.) This can be seen in the Flesch-Kincaid Reading ease score (**FRE**) for the corpus which, for the uncompressed examples, is only 10.29 (whereas for legislation as a whole we would expect a figure of around 40). That is, in only selecting provisions which are candidates for compression, those provisions are naturally more complex (have a higher FRE score) than legislative text as a whole. The FRE score for the compressed examples is 39.23.

## Parsed data

