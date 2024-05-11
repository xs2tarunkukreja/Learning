# Working with Text Data
Text is a sequence data. Sequence of characters. Sequence of Words. Words are mainly used.

Natural Language Understanding - Document classification; Sentiment Analysis; Authon Identification; QnA.

No model/computer understand text in human sense.

Deep Leaning is just find simple pattern to word, sentence or paragraphs.
It doesn't take text as input, they only work with numeric tensors.

Vectorizing Text is a process of tranforming text to numeric tensors.
    By segmenting text into words and transfer each word into vector.
    By segmention text into character and transforming each character into vector.
    By extracting "N-grams"of words or characters and transforming each N-gram into a vector. N-grams are overlapping group of multiple consecutive words or characters.

Different unit to break text (char, word, n-gram) is called token. Breaking text to such token is called tokenization.

Text > Tokens > associated vectors > Sequence Tensors.

Two hot techniques to associate vector with token:
    One Hot Encoding
    token Embedding
## One Hot Encoding
Associate an unique index with each word, then turn this integer index i into a binary vector of N (Size of vocabulary) size to 1. other are 0.

from keras.preprocessing.text import Tokenizer

one-hot hashing trick - If unique token are 2 large in count.
    light weight hash function which map word to index. 
    We don't keep mapping of word with index.
    Issue - hash collision.

## Token Embedding - Word Embedding
Dense Vector.
One Hot Encoding - are binary, sparse, very high dimension.
This is learned from data.
Two ways -
    Pre-trained embedding. A model is used for word embedding which is pre-trained.
    Model first task is embedding and then main task.
### Learn word embedding with embedding layer
Pick the vector at random. Resulting embedding space have no structure.
Can't extract any relationship between word based on relationship between vectors.

L2 distance is not depend on meaning.

from keras.layers import Embedding
// The Embedding layer takes at least two arguments:
//      the number of possible tokens, here 1000 (1 + maximum word index),
//      the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)
?

### Using Pre-trained word embedding
## N-Grams
2-Gram for "The cat sat" - [The, The cat, cat, cat sat, sat]

# Understanding Recurrent Neural Network

# Advanced Usage of RNN

# Sequence Processing with convnet