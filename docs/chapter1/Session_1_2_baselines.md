# Baseline with Regexes and spaCy for Spam Detection

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_2_baselines.ipynb)


In this notebook, we will:

1. Load a spam detection dataset from Hugging Face.
2. Split our data into **train**, **dev**, and **test** sets, and explain why we need all three.
3. Create a **regex-based baseline pipeline**:
    * Build naive patterns from the **train** set.
    * Evaluate on **test** set.
    * Check results on **dev** set to find false positives/negatives.
    * Update regex rules.
    * Final metrics on the **test** set.
4. Build a **spaCy pipeline** for spam detection:
    * Use token and phrase matchers.
    * Repeat the same steps (train -> dev -> refine -> test).
5. Compare results between the improved regex approach and spaCy approach.

---

## Setup and Imports

We’ll need:
- **datasets**: To load the spam dataset.
- **scikit-learn**: For splitting the dataset and computing metrics.
- **re** (built-in): For regex-based matching.
- **spaCy**: For token and phrase matchers.

Make sure to look at this [link](../notebooks.md#setup-instructions) to install all the dependencies.

```python
# If you're in a local environment, uncomment the lines below:
# !poetry run python -m spacy download en_core_web_sm

import re
import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the small English model for spaCy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # Increase max length to handle large texts if needed
```

---

## 1. Load the Dataset

We'll use [NotShrirang/email-spam-filter](https://huggingface.co/datasets/NotShrirang/email-spam-filter). It's a dataset with email text labeled as spam or not spam.

```python
# Load the spam detection dataset from Hugging Face.
dataset = load_dataset("NotShrirang/email-spam-filter")
dataset
```

We expect the dataset to have a `train` split by default, which we’ll further split into **train**, **dev**, and a final **test**. Alternatively, we can keep the existing train as a larger pool and create dev/test from it. Some datasets also come with separate test sets. We'll check what's available after loading.

```python
# Check the dataset's column structure.
dataset["train"].features
```

---

## 2. Create Train/Dev/Test Splits

**Why do we need a dev set in addition to a train/test set?**

- **Train** set: used to fit our model (or in this case, develop our regex/spaCy patterns).
- **Dev** (validation) set: used to **tweak** or **refine** patterns, hyperparameters, etc., without touching the final test. This prevents overfitting on the test set.
- **Test** set: final unbiased evaluation.

If we only had train/test, we might continually adjust our method to do better on the test set, inadvertently tuning to that test distribution. The dev set helps keep the test set "truly" unseen.

```python
# Convert the Hugging Face dataset split to a pandas DataFrame.
df_data = dataset["train"].to_pandas()
df_data.head()
```

```python
# Split the single 'train' dataset into 60/20/20 (train/dev/test) using stratification to keep the label distribution.
df_train, df_temp = train_test_split(df_data, test_size=0.4, stratify=df_data["label"], random_state=42)
df_dev, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42)

# Print the sizes of each split.
print("Train size:", len(df_train))
print("Dev size:  ", len(df_dev))
print("Test size: ", len(df_test))
```

Now we have 3 separate splits. We'll define some helper functions for evaluation.

```python
def compute_metrics(y_true, y_pred):
    # Compute accuracy, precision, recall, and F1-score using scikit-learn.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def print_metrics(metrics_dict, prefix=""):
    # Print metrics with a given prefix.
    print(f"{prefix} Accuracy:  {metrics_dict['accuracy']*100:.2f}%")
    print(f"{prefix} Precision: {metrics_dict['precision']*100:.2f}%")
    print(f"{prefix} Recall:    {metrics_dict['recall']*100:.2f}%")
    print(f"{prefix} F1-score:  {metrics_dict['f1']*100:.2f}%\n")
```

---

## 3. Regex-Based Baseline

### 3a. Create a first naive pipeline

We’ll look at the **train set** to find some potential spam indicators. Typically, spam might have words like `free`, `win`, `urgent`, `congratulations`, etc. This is just a guess. In a real scenario, you’d examine the train data more carefully.

```python
import collections

# Select texts for spam and ham emails.
spam_texts = df_train[df_train["label"] == "spam"]["text"].values
ham_texts = df_train[df_train["label"] == "ham"]["text"].values

def tokenize(text):
    # Tokenize text by extracting words, lowercasing.
    return re.findall(r"\w+", text.lower())

# Collect words from spam emails.
spam_words = []
for txt in spam_texts:
    spam_words.extend(tokenize(txt))

spam_counter = collections.Counter(spam_words)
spam_most_common = spam_counter.most_common(20)
spam_most_common
```

We clearly see a lot of common words in the spam emails. "The", "of", ... stop words in English. Let's get rid of them. I imagine there are a lot of numbers and punctuation as well. Let's get rid of them too.

```python
from spacy.lang.en.stop_words import STOP_WORDS
import string

punctuation = string.punctuation
numbers = string.digits

stop_words = set(STOP_WORDS)

# Filter spam words: remove stop words, punctuation, digits and words of length <= 3.
spam_words = []
for txt in spam_texts:
    for word in tokenize(txt):
        if word not in stop_words and word not in punctuation and word not in numbers and len(word) > 3:
            spam_words.append(word)

spam_counter = collections.Counter(spam_words)
spam_most_common = spam_counter.most_common(20)
spam_most_common
```

We'll pick a few frequent tokens as naive spam triggers. (In reality, you'd do more thorough exploration or use a more advanced approach—but let's keep it simple for demonstration.)

```python
# Define a basic regex pattern that flags emails containing typical spammy words.
spam_keywords = ["free", "http", "www", "money",
                 "win", "winner", "congratulations",
                 "urgent", "claim", "prize", "click",
                 "price"]
# Compile the regex pattern with ignore case.
pattern = re.compile(r"(" + "|".join(spam_keywords) + r")", re.IGNORECASE)

def regex_spam_classifier(text):
    # If any spam keyword is found, classify as spam (1), otherwise not spam (0).
    if pattern.search(text):
        return 1  # spam
    return 0     # not spam
```

---

### 3b. Get metrics on the **test** set

Even though we said we’d refine on dev, let’s see how it does out-of-the-box on the test set. (Sometimes it’s informative to check a naive baseline right away.)

```python
# Get true labels and predict using the regex classifier.
y_test_true = df_test["label_num"].values
y_test_pred = [regex_spam_classifier(txt) for txt in df_test["text"].values]

# Compute and print metrics on the test set.
test_metrics = compute_metrics(y_test_true, y_test_pred)
print_metrics(test_metrics, prefix="Regex Baseline (Test) ")
```

---

Okay, not so bad, we get 70% of the spam emails, but we also have a lot of false positives—almost 50% of our predictions are false positives!!

---

### 3c. Check dev set, find false positives & negatives

Let’s see how many spam messages were missed (false negatives) and how many ham messages were flagged as spam (false positives) on the dev set.

```python
# Get true labels and texts for the dev set.
y_dev_true = df_dev["label_num"].values
texts_dev = df_dev["text"].values

# Predict on the dev set.
y_dev_pred = [regex_spam_classifier(txt) for txt in texts_dev]
dev_metrics = compute_metrics(y_dev_true, y_dev_pred)
print_metrics(dev_metrics, prefix="Regex Baseline (Dev) ")

# Identify false positives (ham predicted as spam) and false negatives (spam predicted as ham).
fp_indices = []  # false positives indices
fn_indices = []  # false negatives indices

for i, (gold, pred) in enumerate(zip(y_dev_true, y_dev_pred)):
    if gold == 0 and pred == 1:
        fp_indices.append(i)
    elif gold == 1 and pred == 0:
        fn_indices.append(i)

print("False Positives:", len(fp_indices), "examples")
print("False Negatives:", len(fn_indices), "examples")
```

First thing is that the metrics are quite similar from the test set. Which means that both sets may be similar. Therefore if we find a way to improve on the dev set, we should see an improvement on the test set.

We clearly see that we have a lot of false positives, also a significant number of false negatives. Therefore first, we may want to cover more cases and then create some other rules to reduce the number of false positives.

---

### 3d. Analyze FN to improve regex

Let's first take a look at the false negatives to see if we can improve the regex.

```python
print("\n--- Some False Negatives ---\n")
for idx in fn_indices[:20]:
    print("DEV INDEX:", idx)
    print(texts_dev[idx][:300], "...")
    print("---")
```

Okay, looks interesting, maybe let's look for words that appear in the false negatives but not in the false positives.

```python
# Let's look for words that appear a lot in the false negatives but not so much in the false positives.
# We will count the words in false negatives and false positives after filtering out stop words, punctuation, and numbers.

fn_words = []
for idx in fn_indices:
    for word in tokenize(texts_dev[idx]):
        if word not in stop_words and word not in punctuation and word not in numbers and len(word) > 3:
            fn_words.append(word)

fp_words = []
for idx in fp_indices:
    for word in tokenize(texts_dev[idx]):
        if word not in stop_words and word not in punctuation and word not in numbers and len(word) > 3:
            fp_words.append(word)

fn_counter = collections.Counter(fn_words)
fp_counter = collections.Counter(fp_words)

# Create a ratio of occurrences in the false negatives over the total occurrences.
fn_ratio = {word: fn_counter.get(word, 0) / (fp_counter.get(word, 0) + fn_counter.get(word, 0))
            for word in fn_counter if fp_counter.get(word, 0) + fn_counter.get(word, 0) > 4}

# Sort words by ratio.
fn_ratio = sorted(fn_ratio.items(), key=lambda x: x[1], reverse=True)

# Print the top words (appear more in false negatives than false positives).
for word, ratio in fn_ratio[:50]:
    print(word, ratio)
```

Well looks like we have some interesting words there. Let's add them to the regex. We do it a dumb way here, but in practice we should explore a bit more.

```python
# Update the spam keywords list by adding new words observed in false negatives.
spam_keywords = ["free", "http", "www", "money",
                 "win", "winner", "congratulations",
                 "urgent", "claim", "prize", "click",
                 "price", "viagra", "vialium", "medication",
                 "aged", "xana", "xanax", "asyc", "cheap",
                 "palestinian", "blood", "doctor", "cialis",
                 "minutes", "vicodin", "soft", "loading",
                 "csgu", "medications", "prescription", "spam", "stop"]
pattern = re.compile(r"(" + "|".join(spam_keywords) + r")", re.IGNORECASE)

def regex_spam_classifier_v0_2(text):
    # Use the updated regex pattern.
    if pattern.search(text):
        return 1  # spam
    return 0     # not spam
```

```python
# Evaluate the updated regex classifier on the test set.
y_test_true = df_test["label_num"].values
y_test_pred = [regex_spam_classifier_v0_2(txt) for txt in df_test["text"].values]

test_metrics = compute_metrics(y_test_true, y_test_pred)
print_metrics(test_metrics, prefix="Regex Baseline (Test) ")
```

Incredible, meaning that just by adding a few words we get a huge improvement in the metrics (+10% of recall!) and the precision is still more or less the same.

---

### 3e. Analyze FP to improve regex

Let's do the same for the false positives. Meaning that we will find words that appear a lot in the false positives but not so much in the false negatives.
If the message is detected as spam, we will apply another regex to check if it contains any of the words in the false positives. If it does, we will label it as ham.

First let's check the dev set false positives.

```python
# Evaluate on the dev set.
y_dev_true = df_dev["label_num"].values
texts_dev = df_dev["text"].values

y_dev_pred = [regex_spam_classifier_v0_2(txt) for txt in texts_dev]
dev_metrics = compute_metrics(y_dev_true, y_dev_pred)
print_metrics(dev_metrics, prefix="Regex Baseline (Dev) ")

# Identify false positives and false negatives.
fp_indices = []  # false positives: predicted spam but actually ham
fn_indices = []  # false negatives: predicted ham but actually spam

for i, (gold, pred) in enumerate(zip(y_dev_true, y_dev_pred)):
    if gold == 0 and pred == 1:
        fp_indices.append(i)
    elif gold == 1 and pred == 0:
        fn_indices.append(i)

print("False Positives:", len(fp_indices), "examples")
print("False Negatives:", len(fn_indices), "examples")
```

We see that we reduced by two the number of false negatives. Let's see if we can reduce the number of false positives.

```python
print("\n--- Some False Positives ---\n")
for idx in fp_indices[:20]:
    print("DEV INDEX:", idx)
    print(texts_dev[idx][:300], "...")
    print("---")
```

```python
# Now let's analyze words that appear a lot in the false positives but not so much in the false negatives.
# First, collect words in all positive (spam) examples from the dev set.
positive_indices = []
for i, (gold, pred) in enumerate(zip(y_dev_true, y_dev_pred)):
    if gold == 1:
        positive_indices.append(i)

positive_words = []
for idx in positive_indices:
    for word in tokenize(texts_dev[idx]):
        if word not in stop_words and word not in punctuation and word not in numbers and len(word) > 3:
            positive_words.append(word)

fp_words = []
for idx in fp_indices:
    for word in tokenize(texts_dev[idx]):
        if word not in stop_words and word not in punctuation and word not in numbers and len(word) > 3:
            fp_words.append(word)

fp_counter = collections.Counter(fp_words)
positive_counter = collections.Counter(positive_words)

# Create a ratio of occurrences in false positives vs. total occurrences.
fp_ratio = {word: fp_counter.get(word, 0) / (fp_counter.get(word, 0) + positive_counter.get(word, 0))
            for word in fp_counter if fp_counter.get(word, 0) + positive_counter.get(word, 0) > 3}

# Sort and print the words by the ratio.
fp_ratio = sorted(fp_ratio.items(), key=lambda x: x[1], reverse=True)

for word, ratio in fp_ratio[:50]:
    print(word, ratio)
```

A bit less easy, but we can try to create a new regex that should cover the false positives. A lot of names and surnames appear there, maybe quiting them would help. And also some corporate words such as "following" or "brcc".

```python
# Define the spam keywords (same as before) and a new set of ham keywords to help avoid false positives.
spam_keywords = ["free", "http", "www", "money",
                 "win", "winner", "congratulations",
                 "urgent", "claim", "prize", "click",
                 "price", "viagra", "vialium", "medication",
                 "aged", "xana", "xanax", "asyc", "cheap",
                 "palestinian", "blood", "doctor", "cialis",
                 "minutes", "vicodin", "soft", "loading",
                 "csgu", "medications", "prescription", "spam", "stop"]
ham_keywords = ["hillary", "christy", "chapman", "susan", "reinhardt",
                "sweeney", "melissa", "hughes", "lisa", "trisha",
                "september", "tracked", "wellhead", "volumes", "meter",
                "offshore", "county", "manage", "brcc", "ivmh"]

# Compile two patterns: one for spam and one for ham.
pattern_spam_v0_3 = re.compile(r"(" + "|".join(spam_keywords) + r")", re.IGNORECASE)
pattern_ham_v0_3 = re.compile(r"(" + "|".join(ham_keywords) + r")", re.IGNORECASE)

def regex_spam_classifier_v0_3(text):
    # Compare the counts of spam and ham keywords in the text.
    # If more spam keywords than ham keywords are found, classify as spam.
    if len(pattern_spam_v0_3.findall(text)) > len(pattern_ham_v0_3.findall(text)):
        return 1  # spam
    return 0     # not spam
```

---

### 3f. Test on test set

We do the final metrics on the test set now that we have a more refined approach. (Though in practice, you might do multiple dev cycles, carefully checking you’re not overfitting.)

```python
# Evaluate the refined regex classifier on the test set.
y_test_true = df_test["label_num"].values
y_test_pred = [regex_spam_classifier_v0_3(txt) for txt in df_test["text"].values]

test_metrics = compute_metrics(y_test_true, y_test_pred)
print_metrics(test_metrics, prefix="Regex Baseline (Test) ")
```

---

### 3g. Limitations

Clearly, a regex approach is limited. We’ll often get false positives for edge cases or false negatives for spam that doesn’t match our known keywords. Regexes can’t capture synonyms or context. That’s where an ML approach or more advanced text processing can help. But still we get 70% in F1 without any ML or advanced text processing!

---

## 4. spaCy Approach

We’ll create a small spaCy pipeline using the **Matcher** or **TokenMatcher** to detect spammy patterns. This is still rule-based, but spaCy makes it easier to do **token-based** patterns or phrase matching that’s more robust than plain regex.

### 4a. Token matcher

We can define token-based patterns: e.g., if a doc has `[{'LOWER': 'free'}]` or `[{'LOWER': 'click'}, {'LOWER': 'now'}]`.

```python
from spacy.matcher import Matcher

# Initialize the spaCy matcher with the shared vocabulary.
matcher = Matcher(nlp.vocab)

# Define token-level patterns.
pattern_free = [{"LOWER": "free"}]
pattern_click_now = [{"LOWER": "click"}, {"LOWER": "now"}]
pattern_urgent = [{"LOWER": "urgent"}]
# You can define additional patterns as needed.

# Add patterns to the matcher with a unique identifier for each.
matcher.add("FREE", [pattern_free])
matcher.add("CLICK_NOW", [pattern_click_now])
matcher.add("URGENT", [pattern_urgent])
```

---

### 4b. spaCy-based classifier

We'll define a function that processes text with `nlp`, runs the matcher, and if any match is found, we label it spam. We'll refine similarly by analyzing dev set mistakes.

```python
def spacy_matcher_spam(doc):
    # Run the matcher on the spaCy doc.
    matches = matcher(doc)
    if matches:
        return 1  # If any pattern is matched, classify as spam.
    return 0

def spacy_spam_classifier(text):
    # Process the text using spaCy and then classify using the matcher.
    doc = nlp(text)
    return spacy_matcher_spam(doc)
```

---

### 4c. Evaluate on dev set -> refine -> evaluate on test set

Let’s do it quickly, given we already know the general approach. We'll compute dev metrics, see if we can spot improvements, and finalize on test.

```python
# Evaluate the spaCy classifier on the test set.
y_test_pred_spacy = [spacy_spam_classifier(t) for t in df_test["text"].values]
test_metrics_spacy = compute_metrics(y_test_true, y_test_pred_spacy)
print_metrics(test_metrics_spacy, "spaCy Baseline (Test)")
```

In practice, we’d repeat the false positive/negative analysis from earlier. I'll skip it as you can do it yourself :).

---

## 5. Compare Regex vs. spaCy Approaches

We can summarize the final test metrics side by side.

```python
print("--- Final Comparison on Test Set ---\n")
print("Regex v2:")
print_metrics(test_metrics)
print("spaCy v2:")
print_metrics(test_metrics_spacy)
```

We spent different amounts of time on each approach, and that's why the metrics for regexes are better. With spaCy we can do more complex patterns and that's why it's more time consuming to implement. But let's imagine we use both models to see if we can improve the metrics.

To do so let's compare the false positives and false negatives of the two models on the dev set. Maybe there are some patterns that are detected by one model but not by the other one.

```python
# Get predictions on the dev set for both classifiers.
y_dev_pred_spacy = [spacy_spam_classifier(t) for t in df_dev["text"].values]
y_dev_pred_regex = [regex_spam_classifier_v0_3(t) for t in df_dev["text"].values]

# Collect indices for false positives and false negatives for spaCy.
fp_indices_spacy = []
fn_indices_spacy = []

for i, (gold, pred) in enumerate(zip(y_dev_true, y_dev_pred_spacy)):
    if gold == 0 and pred == 1:
        fp_indices_spacy.append(i)
    elif gold == 1 and pred == 0:
        fn_indices_spacy.append(i)

# Collect indices for false positives and false negatives for regex.
fp_indices_regex = []
fn_indices_regex = []

for i, (gold, pred) in enumerate(zip(y_dev_true, y_dev_pred_regex)):
    if gold == 0 and pred == 1:
        fp_indices_regex.append(i)
    elif gold == 1 and pred == 0:
        fn_indices_regex.append(i)
```

Now let's look at the intersection of the two sets.

```python
common_fp = set(fp_indices_spacy) & set(fp_indices_regex)
common_fn = set(fn_indices_spacy) & set(fn_indices_regex)

print("Models:\t spaCy\t regex")
print("False Positives:\t", len(fp_indices_spacy), "\t", len(fp_indices_regex))
print("False Negatives:\t", len(fn_indices_spacy), "\t", len(fn_indices_regex))
print("Common False Positives:\t", len(common_fp))
print("Common False Negatives:\t", len(common_fn))
```

Therefore we see that the whole false negatives from regex are detected by spaCy. But there are less false positives from spaCy. Maybe adding the spaCy patterns to confirm false positives from regex would help. This is something you can test when you have optimized the spaCy patterns and even use a model that could learn how much weight to give to each model. Or just a statistical weight to avoid using Machine Learning models!
