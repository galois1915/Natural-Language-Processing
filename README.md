# Natural-Language-Processing
In the recent years, Natural Language Processing (NLP) has experienced fast growth primarily due to the performance of the language models:
* GPT-3
* BERT

Classical NLP architectures:
* Bag-of-words (BoW)
* Word embeddings
* Recurrent neural networks 
* Generative networks.

Task in NLP:
* Text classification
* Intent classification
* Sentiment analysis
* Named Entity Recognition
* Keyword extraction
* Text Summarization
* Question/Answer

If we want to solve Natural Language Processing (NLP) tasks with neural networks, we need some way to represent text as tensors. Computers already represent textual characters as numbers that map to fonts on your screen using encodings such as ASCII or UTF-8.

We can use different approaches when representing text:
* **Character-level representation**: Each letter would correspond to a tensor column in one-hot encoding.
* **Word-level representation** when we create a vocabulary of all words in our text sequence or sentence(s), and then represent each word using one-hot encoding.

To unify those approaches, we typically call an atomic piece of text a **token**. In some cases tokens can be letters, in other cases - words, or parts of words. The process of converting text into a sequence of tokens is called **tokenization**. Next, we need to assign each token to a number, which we can feed into a neural network. This is called **vectorization**, and is normally done by building a token vocabulary.

Datasets:
* AG_NEWS
* IMDB

## Bag of Words and TF-IDF
## Word embeddings
## Recurrent Neural Networks
## Generate text with RNN
## Pre-trained models