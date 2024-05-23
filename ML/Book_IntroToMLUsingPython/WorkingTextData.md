# Working with Text Data
Continuous feature and categorical features.
Third kind of feature - items from a fixed list. 'Text'

Email is legitimate email or span.
Message is query or complaint.

We need to process Text data before we apply it to ML algorithm.

# Types of Data Represented as String
Text is usually just a string in your dataset but not all string are treated as Text feature.
Some are categorical variables.

4 Kinds of String Data -
    Categorical - Fix list of strings.
    Free string that can be semantically mapped to categories - Same as first. But instead of fix list, user type..
    Structure string data - E.g. phone number. It need Domain Knowledge.
    Text data

# Example Application: Sentiment Analysis of Movie Review
There is a helper function in scikit-learn to load files stored in such a folder structure, where each subfolder corresponds to a label, called load_files.
    train\pos
    train\neg
    test\pos
    test\neg

from sklearn.datasets import load_files
reviews_train = load_files("data/aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target

Review contains HTML formatting, so we should clean this as well.
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

Binary Classification for Movie Review.

ML can't understand Text data.

# Representing Text Data as a Bag of Words
Corpus = Text String

When using this representation, we discard most of the structure of the input text, like chapters, paragraphs, sentences, and formatting, and only count how often each word appears in each text in the corpus.

Three Steps -
    Tokenization - Split each document into words that appear in it (called token).
    Vocabulary Building - Collect the vocabulary of all words that appear in any of the document and number them.
    Encoding - For each document, count how often each words in the vocabulary appear in this document.

Output in vector of word count for each document.

Here the order of word is completly irrelevant.

bards_words =["The fool doth think he is wise,", "but the wise man knows himself to be a fool"]
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)

vect.vocabulary_ // 13 words.

bag_of_words = vect.tranform(bards_words) // 2*13
// Sparse matrix - Only store the entires of non-zeros.
// We can convert sparse matrix to dense Numpy Array
bag_of_words.toarray()

## Moview Review 
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train) // 25,000×74,849, indicating that the vocabulary contains 74,849 entries.

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)

// Improve C - regularization parameter.
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

grid.best_score_ // 0.89
grid.best_params // {'C':0.1}

## Improve Extraction of Words
Default regular expression - "\b\w\w+\b"
One way to cut back on these is to only use tokens that appear in at least two documents

vect = CountVectorizer(min_df=5).fit(text_train) // 5 is word should be in 5 document atleast.
X_train = vect.transform(text_train)

# Stopword
Another way that we can get rid of uninformative words is by discarding words that are too frequent to be informative. There are two main approaches: using a language specific list of stopwords, or discarding words that appear too frequently.

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Rescaling of Data with tf-idf
Instead of dropping features that are deemed unimportant, another approach is to rescale features by how informative we expect them to be. One of the most common ways to do this is using the term frequency–inverse document frequency (tf–idf) method.

Give high weight to any term that appears often in a particular document, but not in many documents in the corpus.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None), LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)

It is a purely unsupervised technique.

high-tf–idf features terms only appear in reviews for this particular show or franchise, but tend to appear very often in these particular reviews. This is very clear, for example, for "pokemon", "smallville", and "doodlebops", but "scanners" here actually also refers to a movie title. These words are unlikely to help us in our sentiment classification task.

# Investigation Model Coefficient
The 25 largest and 25 smallest coefficients of the logistic regression model.
grid.best_estimator_.named_steps["logisticregression"].coef_

-ve coefficient means negative review.

# Bag of words with More than One word (n-Grams)
Bag of words discard the order of words completely.
the two strings - “it’s bad, not good at all” and “it’s good, not bad at all” 
have exactly the same representation, even though the meanings are inverted.

The counts of pairs or triplets of tokens that appear next to each other.
Pair of tokens are bigrams.
Triplets are trigrams.
n-grams.

Changing the ngram_range parameter of CountVectorizer or TfidfVectorizer.

cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
Vocabulary:
['be fool', 'but the', 'doth think', 'fool doth', 'he is', 'himself to', 'is wise', 'knows himself', 'man knows', 'the fool', 'the wise', 'think he', 'to be', 'wise man']

For most applications, the minimum number of tokens should be one, as single words often capture a lot of meaning. Adding bigrams helps in most cases. Adding longer sequences—up to 5-grams—might help too, but this will lead to an explosion of the number of features and might lead to overfitting.

# Advanced Tokenization, stemming, and Lemmatization
For the purposes of a bag-of-words model, the semantics of "drawback" and "drawbacks" are so close that distinguishing them will only increase overfitting.
Another example - different verb forms and a noun relating to the verb.

This problem can be overcome by representing each word using its word stem, which involves identifying (or conflating) all the words that have the same word stem. If this is done by using a rule-based heuristic, like dropping common suffixes, it is usually referred to as stemming.

If instead a dictionary of known word forms is used (an explicit and human-verified system), and the role of the word in the sentence is taken into account, the process is referred to as lemmatization and the standardized form of
the word is referred to as the lemma. Both processing methods, lemmatization and stemming, are forms of normalization that try to extract some normal form of a word.

import spacy
import nltk

en_nlp = spacy.load('en')
stemmer = nltk.stem.PorterStemmer()

def compare_normalization(doc):
    // tokenize document in spacy
    doc_spacy = en_nlp(doc)
    // print lemmas found by spacy
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    // print tokens found by Porter stemmer
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

compare_normalization(u"Our meeting today was worse than yesterday, ""I'm scared of meeting the clients tomorrow.")

Lemmatization:
['our', 'meeting', 'today', 'be', 'bad', 'than', 'yesterday', ',', 'i', 'be', 'scared', 'of', 'meet', 'the', 'client', 'tomorrow', '.']
Stemming:
['our', 'meet', 'today', 'wa', 'wors', 'than', 'yesterday', ',', 'i', "'m", 'scare', 'of', 'meet', 'the', 'client', 'tomorrow', '.']


Stemming is always restricted to trimming the word to a stem, so "was" becomes "wa".
Lemmatization can retrieve the correct base verb form, "be".

Lemmatization is better normalization then Stemming.

# Topic Modeling and Document Clustering
Topic modeling, which is an umbrella term describing the task of assigning each document to one or multiple
topics, usually without supervision. 
Example - news to politics, sports, finance etc.

## Latent Dirichlet Allocation
Find groups of words (the topics) that appear together frequently.

Remove common occuring words
vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10, learning_method="batch", max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)