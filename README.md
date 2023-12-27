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

Datasets:
* AG_NEWS
* IMDB

If we want to solve Natural Language Processing (NLP) tasks with neural networks, we need some way to represent text as tensors. Computers already represent textual characters as numbers that map to fonts on your screen using encodings such as ASCII or UTF-8.

We can use different approaches when representing text:
* **Character-level representation**: Each letter would correspond to a tensor column in one-hot encoding.
* **Word-level representation** when we create a vocabulary of all words in our text sequence or sentence(s), and then represent each word using one-hot encoding.

To unify those approaches, we typically call an atomic piece of text a **token**. In some cases tokens can be letters, in other cases - words, or parts of words. The process of converting text into a sequence of tokens is called **tokenization**. Next, we need to assign each token to a number, which we can feed into a neural network. This is called **vectorization**, and is normally done by building a token vocabulary. For this we can use this function in <code>PyTorch</code>:
* <code>torchtext.data.utils.get_tokenizer</code>
* <code>torchtext.vocab.vocab</code>
* <code>collection.Count</code>
* encode and decode with <code>vocab.get_stoi()[string]</code>

Now, to build text classification model, we need to feed the whole sentence (or whole news article) into a neural network. The problem here is that each article/sentence has variable length; and all fully-connected or convolution neural networks deal with fixed input size. There are two ways we can handle this problem:
* Find a way to collapse a sentence into fixed-length vector ***Bag-of-Words and TF-IDF representations**.
* Design special neural network architectures that can deal with variable length sequences ***Recurrent neural networks (RNN)**.

## Bag of Words and TF-IDF
### Bag of Words (BoW)
Vector representation is the most commonly used traditional vector representation. Each word is linked to a vector index, vector element contains the number of occurrences of a word in a given document.

In BoW representation, word occurrences are evenly weighted, regardless of the word itself. However, it is clear that frequent words, such as *'a'*, *'in'*, *'the'* etc. are much less important for the classification, than specialized terms. In fact, in most NLP tasks some words are more relevant than others.

### TF-IDF
Stands for **term frequency–inverse document frequency**. It is a variation of bag of words, where instead of a binary 0/1 value indicating the appearance of a word in a document, a floating-point value is used, which is related to the **_frequency of word occurrence_** in the corpus.

The formula to calculate TF-IDF is:  $w_{ij} = tf_{ij}\times\log({N\over df_i})$

Here's the meaning of each parameter in the formula:
* $i$ is the word 
* $j$ is the document
* $w_{ij}$ is the weight or the importance of the word in the document
* $tf_{ij}$ is the number of occurrences of the word $i$ in the document $j$, i.e. the BoW value we have seen before
* $N$ is the number of documents in the collection
* $df_i$ is the number of documents containing the word $i$ in the whole collection

TF-IDF value $w_{ij}$ increases proportionally to the number of times a word appears in a document and is offset by the number of documents in the corpus that contains the word, which helps to adjust for the fact that some words appear more frequently than others. For example, if the word appears in *every* document in the collection, $df_i=N$, and $w_{ij}=0$, and those terms would be completely disregarded.

Even though TF-IDF representation calculates different weights to different words according to their importance, it is unable to correctly capture the meaning, largely because the order of words in the sentence is still not taken into account.

## Word embeddings
## Recurrent Neural Networks
## Generate text with RNN
## Pre-trained models