# ML with TF-IDF + Logistic Regression on IMDB Dataset

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_3_tfidf.ipynb)


In this notebook, we will:

0. Set a NumPy random seed and load the **IMDB** dataset from Hugging Face, sampling 8k examples for training and 3k for testing. We will further split the training set into **train** and **dev**.
1. Tackle sentiment classification with **TF-IDF** features and **Logistic Regression**.
2. Explore key hyperparameters: `stop_words`, `tokenizer`, `analyzer`, `min_df`, `max_df`, `ngram_range`, `max_features`, and `vocabulary` with clear examples.
3. Create a scikit-learn **Pipeline** with a pre-processing class (if needed), TF-IDF vectorizer, and Logistic Regression (with comments on regularization parameters and class weights) and evaluate the results on the **dev** and **test** sets.
4. Analyze feature importance:
    - a. Look at the contribution of each token.
    - b. Show (with color highlights) which words drive the predictions.
    - c. Rank the top words for positive and negative classes.
5. Update the pre-processing pipelines based on this analysis.

---

## 0. Load the IMDB dataset

### What is the IMDB Dataset?

The IMDB dataset is a widely-used benchmark in sentiment analysis. It consists of movie reviews collected from the Internet Movie Database (IMDB), each labeled with a binary sentiment indicating whether the review is **positive** or **negative**.

#### Key Features:
- **Text Reviews:** Each record is a movie review written in natural language.
- **Binary Sentiment Labels:** Reviews are marked as `0` (negative) or `1` (positive).
- **Benchmark Dataset:** It is extensively used for training and evaluating models in natural language processing, particularly for sentiment classification tasks.

I set a random seed for reproducibility. I sample 8000 examples for training and 3000 for testing. I further split the training set into train and dev (80/20 split).

```python
# 0. Setup & Data Loading
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Set numpy random seed for reproducibility
np.random.seed(42)

# Load the IMDB dataset from Hugging Face
imdb = load_dataset("imdb")

# Get the full train and test splits
imdb_train_full = imdb["train"]
imdb_test_full = imdb["test"]

# Convert to pandas DataFrames
train_df = pd.DataFrame(imdb_train_full)
test_df = pd.DataFrame(imdb_test_full)

# Sample 8000 examples from train and 3000 from test
train_df = train_df.sample(n=8000, random_state=42).reset_index(drop=True)
test_df = test_df.sample(n=3000, random_state=42).reset_index(drop=True)

# Split train_df into train and dev (80/20 split) using stratification on the label
train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))
```

---

## 1. TF-IDF + Logistic Regression for Sentiment Classification

We will build a machine learning model using TF-IDF features combined with Logistic Regression to predict sentiment (positive/negative) from IMDB reviews.

As mentioned in the slides, we can use the `TfidfVectorizer` to convert the text data into a matrix of TF-IDF features.

We can then use a `LogisticRegression` model to predict the sentiment of the reviews. It's a simple model that is easy to understand and interpret.

We can also use a `Pipeline` to streamline the process of converting the text data into TF-IDF features and then using the Logistic Regression model to predict the sentiment. `Pipeline` is a powerful tool that allows us to chain together multiple steps of the process into a single object.

```python
# Import necessary libraries for TF-IDF and Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import FunctionTransformer
import re

# For demonstration, we define a very simple text preprocessor (could be extended)
def simple_preprocessor(text):
    # Lowercase the text and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Example hyperparameter settings for TfidfVectorizer
tfidf_params = {
    "stop_words": "english",           # Remove English stop words
    "tokenizer": None,                 # Use default tokenizer; can be replaced by a custom one
    "analyzer": "word",                # Analyze words (as opposed to characters)
    "min_df": 5,                       # Ignore terms that appear in fewer than 5 documents
    "max_df": 0.8,                     # Ignore terms that appear in more than 80% of the documents
    "ngram_range": (1, 1),             # Unigrams by default (can change to (1,2) for bigrams, etc.)
    "max_features": 10000              # Limit vocabulary size
}

# Create the pipeline with three steps: preprocessor, TF-IDF vectorizer, and Logistic Regression.
pipeline = Pipeline([
    ("preprocessor", FunctionTransformer(lambda X: [simple_preprocessor(text) for text in X])),
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("logreg", LogisticRegression(
         # Regularization: C is the inverse of regularization strength (smaller values specify stronger regularization)
         # class_weight: 'balanced' automatically adjusts weights inversely proportional to class frequencies
         C=1.0,
         penalty='l2',
         solver='lbfgs',
         max_iter=1000,
         class_weight='balanced'
     ))
])

# Display the pipeline steps (for reference)
pipeline
```

---

## 2. Hyperparameter Examples for TF-IDF

Below are some key parameters for the TF-IDF vectorizer with brief explanations:

- **stop_words**: Words to remove (e.g., "english" removes common English stop words).
- **tokenizer**: Function to split text into tokens. You can provide your own function.
- **analyzer**: Determines whether the analyzer operates on word or character level.
- **min_df**: Minimum document frequency; ignore terms that appear in fewer documents.
- **max_df**: Maximum document frequency; ignore terms that appear in a large proportion of documents.
- **ngram_range**: The range of n-values for different n-grams to be extracted (e.g., (1,2) for unigrams and bigrams).
- **max_features**: Maximum number of features (vocabulary size) to consider.
- **vocabulary**: Optionally, you can pass a custom vocabulary.

Below are examples of how those parameters affect the feature matrix.

### Stop_words
**Description:** Removes common words from the text.
**Example:** Remove common English stop words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Using built-in stop words for English
vectorizer1 = TfidfVectorizer(stop_words="english")
vectorizer2 = TfidfVectorizer(stop_words=None)

documents = ["This is a sample document.", "Another document with more text."]
tfidf_matrix1 = vectorizer1.fit_transform(documents)
tfidf_matrix2 = vectorizer2.fit_transform(documents)

print("Features with stop_words='english':", vectorizer1.get_feature_names_out())
print("Features without stop_words:", vectorizer2.get_feature_names_out())
```

### Tokenizer
**Description:** Function to split text into tokens.
**Example:** Use a custom tokenizer that removes punctuation and splits by whitespace.

```python
def custom_tokenizer(text):
    # Remove punctuation and split by whitespace
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().split()

vectorizer1 = TfidfVectorizer(tokenizer=custom_tokenizer)
vectorizer2 = TfidfVectorizer(tokenizer=None)  # Using default tokenizer

documents = ["Hello, world! This is an example. I'm happy to see you.", "Custom tokenizer works well."]
tfidf_matrix1 = vectorizer1.fit_transform(documents)
tfidf_matrix2 = vectorizer2.fit_transform(documents)

print("Tokens using custom tokenizer:", vectorizer1.get_feature_names_out())
print("Tokens without custom tokenizer:", vectorizer2.get_feature_names_out())
```

### Analyzer
**Description:** Determines whether the analyzer operates on word or character level.
**Example:** Analyze text at the character level and at the word level.

```python
# Use character analyzer (each character n-gram will be a feature)
vectorizer_char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
# Use word analyzer with bigrams
vectorizer_word = TfidfVectorizer(analyzer="word", ngram_range=(2, 4))

documents = ["This is a sample document.", "Another document with more text."]
tfidf_matrix_char = vectorizer_char.fit_transform(documents)
tfidf_matrix_word = vectorizer_word.fit_transform(documents)

print("Character n-grams as features:", vectorizer_char.get_feature_names_out())
print("Word n-grams as features:", vectorizer_word.get_feature_names_out())
```

### Min_df and Max_df
**Description:** Minimum and maximum document frequencies for terms to be included in the vocabulary.
**Example:** Include terms that appear in at least 2 documents and at most 80% of the documents.

```python
vectorizer0 = TfidfVectorizer()
vectorizer1 = TfidfVectorizer(min_df=2)
vectorizer2 = TfidfVectorizer(max_df=0.8)

documents = ["This is a sample document.", "Another document with more text.", "This is another document."]
tfidf_matrix0 = vectorizer0.fit_transform(documents)
tfidf_matrix1 = vectorizer1.fit_transform(documents)
tfidf_matrix2 = vectorizer2.fit_transform(documents)

print("Features without min_df and max_df:", vectorizer0.get_feature_names_out())
print("Features with min_df=2:", vectorizer1.get_feature_names_out())
print("Features with max_df=0.8:", vectorizer2.get_feature_names_out())
```

### N-gram Range
**Description:** The range of n-values for different n-grams to be extracted.
**Example:** Extract unigrams and bigrams versus unigrams, bigrams, and trigrams.

```python
vectorizer1 = TfidfVectorizer(ngram_range=(1, 2))
vectorizer2 = TfidfVectorizer(ngram_range=(1, 3))

documents = ["this is a test", "another test example"]
tfidf_matrix1 = vectorizer1.fit_transform(documents)
tfidf_matrix2 = vectorizer2.fit_transform(documents)

print("Features with ngram_range=(1,2):", vectorizer1.get_feature_names_out())
print("Features with ngram_range=(1,3):", vectorizer2.get_feature_names_out())
```

### Max_features
**Description:** Maximum number of features (vocabulary size) to consider.
**Example:** Limit the vocabulary size to only the top 3 features based on term frequency.

```python
vectorizer = TfidfVectorizer(max_features=3)
documents = ["This is a document with a lot of words", "This is another document with a lot of words"]
tfidf_matrix = vectorizer.fit_transform(documents)
print("Features with max_features=3:", vectorizer.get_feature_names_out())
```

### Vocabulary
**Description:** Optionally, you can pass a custom vocabulary.
**Example:** Use a custom vocabulary to restrict features.

```python
custom_vocab1 = {"data": 0, "science": 1, "machine": 2, "learning": 3}
custom_vocab2 = {"data": 0, "science": 1, "machine": 2, "is": 3, "amazing": 4}

vectorizer1 = TfidfVectorizer(vocabulary=custom_vocab1)
vectorizer2 = TfidfVectorizer(vocabulary=custom_vocab2)

documents = ["data science is amazing", "machine learning is part of data science"]
tfidf_matrix1 = vectorizer1.fit_transform(documents)
tfidf_matrix2 = vectorizer2.fit_transform(documents)

print("Features using custom vocabulary:", vectorizer1.get_feature_names_out())
print("Features using custom vocabulary:", vectorizer2.get_feature_names_out())
```

---

## 3. Train the Model and Evaluate

We now train our pipeline on the training set, evaluate on the dev set, and finally check performance on the test set.

You'll see below that this is straightforward as we just need to call the `fit` method on the training set and then the `predict` method on the dev and test sets.

```python
# Train the model
pipeline.fit(train_df["text"], train_df["label"])

# Evaluate on dev set
dev_preds = pipeline.predict(dev_df["text"])
dev_accuracy = accuracy_score(dev_df["label"], dev_preds)
dev_precision = precision_score(dev_df["label"], dev_preds)
dev_recall = recall_score(dev_df["label"], dev_preds)
dev_f1 = f1_score(dev_df["label"], dev_preds)

print("Dev Set Metrics:")
print(f"Accuracy: {dev_accuracy*100:.2f}%")
print(f"Precision: {dev_precision*100:.2f}%")
print(f"Recall: {dev_recall*100:.2f}%")
print(f"F1-score: {dev_f1*100:.2f}%")

# Evaluate on test set
test_preds = pipeline.predict(test_df["text"])
test_accuracy = accuracy_score(test_df["label"], test_preds)
test_precision = precision_score(test_df["label"], test_preds)
test_recall = recall_score(test_df["label"], test_preds)
test_f1 = f1_score(test_df["label"], test_preds)

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy*100:.2f}%")
print(f"Precision: {test_precision*100:.2f}%")
print(f"Recall: {test_recall*100:.2f}%")
print(f"F1-score: {test_f1*100:.2f}%")
```

---

Results are quite good if we compare to the random baseline (50% F1). We outperform the baseline by a large margin. We also see that the results of precision and recall are quite good around 85% without a clear discrepancy between the two.

Let's look at the feature importance analysis to understand which tokens drive the predictions and maybe find some interesting patterns that will help us improve the pipeline.

---

## 4. Feature Importance Analysis

After training, we analyze the learned logistic regression coefficients to understand which tokens drive the predictions.

a. **Contribution of Each Token**: We extract the coefficient for each feature (token).
b. **Visual Examples**: We'll highlight tokens in some example reviews (this example uses HTML formatting for color).
c. **Ranking Top Tokens**: We rank tokens for each class (positive and negative).

```python
import numpy as np
import pandas as pd

# Get the trained Logistic Regression model from the pipeline
logreg = pipeline.named_steps['logreg']
tfidf = pipeline.named_steps['tfidf']

# Get feature names (tokens)
feature_names = np.array(tfidf.get_feature_names_out())

# Logistic Regression coefficients
coefficients = logreg.coef_[0]

# a. Contribution of each token: create a DataFrame to display tokens and their coefficients
coef_df = pd.DataFrame({
    'token': feature_names,
    'coefficient': coefficients
})

# b. Top tokens for positive (assumed label 1) and negative (assumed label 0) sentiment
top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(20)
top_negative = coef_df.sort_values(by='coefficient').head(20)

print("Top Tokens for Positive Sentiment:")
print(top_positive[['token', 'coefficient']])
print("\nTop Tokens for Negative Sentiment:")
print(top_negative[['token', 'coefficient']])

# Plot horizontal bar charts for visualization
from matplotlib import pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].barh(top_positive['token'], top_positive['coefficient'], color='green')
ax[1].barh(top_negative['token'], top_negative['coefficient'], color='red')
ax[0].set_xlabel('Coefficient')
ax[0].set_ylabel('Tokens')
ax[0].set_title('Top Tokens for Positive Sentiment')
ax[1].set_xlabel('Coefficient')
ax[1].set_ylabel('Tokens')
ax[1].set_title('Top Tokens for Negative Sentiment')
plt.tight_layout()
plt.show()
```

We see interesting patterns here. For positive label, the top tokens are mostly related to sentiment. Words like "great", "wonderful", "excellent", "fantastic" are all positive words. For negative label, the top tokens are mostly related to sentiment. Words like "bad", "terrible", "horrible", "awful" are all negative words.

Nevertheless, we see some tokens that are not related to sentiment but are still important for the classification. For example, "today" or "script" are important for the classification but they are not related to sentiment. It looks like the model overfits on those tokens because they may be present too many times in the corpus for positive or negative reviews.

Let's check that hypothesis by looking at the probability to have a positive or negative review depending on the presence of a token.

```python
# Get the TF-IDF vectorizer from the pipeline
tfidf = pipeline.named_steps['tfidf']

# Get the vocabulary and feature names
vocabulary = tfidf.vocabulary_
feature_names = tfidf.get_feature_names_out()

# We just want to know if a word appears or not
# Transform the training data
X_count = tfidf.transform(train_df["text"])

# Convert to binary (1 if word appears, 0 if not)
X_binary = (X_count > 0).astype(int)

# Get the labels
y = train_df["label"].values

# Calculate probabilities
word_sentiment_probs = {}

# Total counts for each sentiment
positive_count = np.sum(y == 1)
negative_count = np.sum(y == 0)
total_count = len(y)

# For each word in the vocabulary
for i, word in enumerate(feature_names):
    # Get documents containing this word
    docs_with_word = X_binary[:, i].toarray().flatten()

    # Count documents with this word for each sentiment
    positive_with_word = np.sum(docs_with_word & (y == 1))
    negative_with_word = np.sum(docs_with_word & (y == 0))
    total_with_word = np.sum(docs_with_word)

    if total_with_word > 0:  # Avoid division by zero
        # P(positive | word)
        p_positive_given_word = positive_with_word / total_with_word

        # P(negative | word)
        p_negative_given_word = negative_with_word / total_with_word

        # Store the probabilities
        word_sentiment_probs[word] = {
            'P(positive|word)': p_positive_given_word,
            'P(negative|word)': p_negative_given_word,
            'count': total_with_word,
            'positive_count': positive_with_word,
            'negative_count': negative_with_word
        }

# Convert to DataFrame for easier analysis
probs_df = pd.DataFrame.from_dict(word_sentiment_probs, orient='index')

# Sort by probability of positive sentiment
most_positive_words = probs_df.sort_values(by='P(positive|word)', ascending=False).head(20)
most_negative_words = probs_df.sort_values(by='P(negative|word)', ascending=False).head(20)

# Display results
print("Words most associated with positive sentiment:")
print(most_positive_words[['P(positive|word)', 'count']])

print("\nWords most associated with negative sentiment:")
print(most_negative_words[['P(negative|word)', 'count']])
```



```python
# Check the sentiment probabilities for specific words
print(word_sentiment_probs["today"])
print(word_sentiment_probs["movie"])
print(word_sentiment_probs["film"])
```

Here we see a lot of words that may not be related to sentiment but are still important for the classification. For example, "tate" or "joss". And we look at the example of "today" that we saw before, the probability to have a positive or negative review depending on the presence of "today" is clearly skewed towards positive reviews. There may be a reason, but this word without context is not a good feature for the classification. Therefore maybe we should remove those words from the vocabulary or use them only with more context (e.g., with higher n-grams).

Let's see another way of looking at the feature importance by highlighting the words in some example reviews.

```python
# c. Example: Highlight words in a review (for illustration, using HTML styling)
# The function below assigns green color to tokens with coefficient > threshold (positive)
# and red color if coefficient < -threshold (negative).
def highlight_review(review, threshold=0.5):
    tokens = review.split()
    highlighted = []
    for token in tokens:
        token_clean = token.lower()
        if token_clean in feature_names:
            # Find index of token in the vocabulary
            idx = np.where(feature_names == token_clean)[0][0]
            coef = coefficients[idx]
            # Color positive words in green and negative words in red
            if coef > threshold:
                token = f'<span style="color:green">{token}</span>'
            elif coef < -threshold:
                token = f'<span style="color:red">{token}</span>'
        highlighted.append(token)
    return ' '.join(highlighted)

# Show highlighted review examples using IPython.display to render HTML
from IPython.display import display, HTML

for i in range(10):
    sample_review = test_df['text'].iloc[i]
    display(HTML(highlight_review(sample_review, threshold=0.1)))
    print('--------------------------------')
```

This is quit interesting to see the results as  we see words like "American", "think" or "right" which are highlighted in red and words like "maybe", "tell" or "film", that are highlighted in green. WE see also a lot of names.

But it looks weird as those words are not related to sentiment. It looks like the model overfits on those tokens.

One potential solution for this problem would be to consider only bigrams or trigrams as features just to increase the context of the tokens. One way would be to use the `ngram_range` parameter in the TF-IDF vectorizer. We could also increase the `min_df` parameter to remove words that are not present in enough documents and reduce words that are present in too many documents such as "film" or "movie".

Let's try to see if this hypothesis is correct by running the pipeline with different stop words. Also we see that the word "like" is in red generally, maybe we should use it with more context ? because like and don't like would dramatically change the sentiment of the review.

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

new_stop_words = ["film", "movie", "american", "think", "right", "maybe", "tell",
                  "couple", "want", "gets", "get", "john", "carter", "rice", "day", "apes",
                  "say"]

for stop_words in [ENGLISH_STOP_WORDS, ENGLISH_STOP_WORDS.union(new_stop_words)]:
    print(f"\nEvaluating with stop_words = {stop_words}")
    tfidf_params["stop_words"] = list(stop_words)
    pipeline.set_params(tfidf=TfidfVectorizer(**tfidf_params))
    pipeline.fit(train_df["text"], train_df["label"])
    preds = pipeline.predict(dev_df["text"])
    acc = accuracy_score(dev_df["label"], preds)
    precision = precision_score(dev_df["label"], preds)
    recall = recall_score(dev_df["label"], preds)
    f1 = f1_score(dev_df["label"], preds)
    print(f"Dev Accuracy: {acc*100:.2f}%")
    print(f"Dev Precision: {precision*100:.2f}%")
    print(f"Dev Recall: {recall*100:.2f}%")
    print(f"Dev F1-score: {f1*100:.2f}%")
```

```python
tfidf = pipeline.named_steps['tfidf']
logreg = pipeline.named_steps['logreg']

vocabulary = tfidf.vocabulary_
feature_names = tfidf.get_feature_names_out()
coefficients = logreg.coef_[0]

def highlight_review(review, threshold=0.5):
    tokens = review.split()
    highlighted = []
    for token in tokens:
        token_clean = token.lower()
        if token_clean in feature_names:
            # Find index of token in the vocabulary
            idx = np.where(feature_names == token_clean)[0][0]
            coef = coefficients[idx]
            # Color positive words in green and negative in red
            if coef > threshold:
                token = f'<span style="color:green">{token}</span>'
            elif coef < -threshold:
                token = f'<span style="color:red">{token}</span>'
        highlighted.append(token)
    return ' '.join(highlighted)



for i in range(10):
    sample_review = test_df['text'].iloc[i]
    display(HTML(highlight_review(sample_review, threshold=0.1)))
    print('--------------------------------')
```

We see that we have removed the words that are not related to sentiment and the model is more focused on the sentiment of the review. But we see others. Maybe we should just limit the vocabulary size to the most important words—the ones that are really important for the classification. This should be investigated.

We see that the model stays with equivalent results. It means that the stop words we removed are not so important for the classification as the model can handle the classification without them.
We would need to investigate further to see if we can improve the results by removing other stop words and feeling confident enough to put such models in production. We don't want the user looking at results and seeing that a word like "film" or "movie" drove the results the user will lose confidence in the model.
