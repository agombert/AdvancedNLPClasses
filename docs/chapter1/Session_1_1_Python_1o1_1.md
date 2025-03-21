
# Python Types

This notebook covers the fundamental and advanced aspects of Python's type system. Understanding types is crucial for writing robust and maintainable code, especially in data science and NLP applications.

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_1_Python_1o1_1.ipynb)

## Table of Contents
1. [Basic Types](#basic)
2. [Advanced Types](#advanced)
3. [Exercises](#exercises)
4. [Real-World Applications](#applications)

---

## 1. Basic Types <a name="basic"></a>

Python is a dynamically typed language, which means you don't need to declare the type of a variable when you create it. The interpreter infers the type based on the value assigned.

### 1.1 Primitive Types

```python
# Integers
age = 25
print(f"age is of type: {type(age)}")  # prints: <class 'int'>

# Floating-point numbers
height = 1.75
print(f"height is of type: {type(height)}")  # prints: <class 'float'>

# Booleans
is_student = True
print(f"is_student is of type: {type(is_student)}")  # prints: <class 'bool'>

# Strings
name = "Alice"
print(f"name is of type: {type(name)}")  # prints: <class 'str'>
```

### 1.2 Complex Types

```python
# Lists - ordered, mutable collections
fruits = ["apple", "banana", "cherry"]
print(f"fruits is of type: {type(fruits)}")  # <class 'list'>

# Tuples - ordered, immutable collections
coordinates = (10.5, 20.8)
print(f"coordinates is of type: {type(coordinates)}")  # <class 'tuple'>

# Dictionaries - key-value pairs
person = {"name": "Bob", "age": 30, "is_student": False}
print(f"person is of type: {type(person)}")  # <class 'dict'>

# Sets - unordered collections of unique elements
unique_numbers = {1, 2, 3, 3, 4, 5}  # Note: duplicates are automatically removed
print(f"unique_numbers is of type: {type(unique_numbers)}")  # <class 'set'>
print(f"unique_numbers contains: {unique_numbers}")  # prints the set
```

### 1.3 Type Conversion

```python
# String to integer
age_str = "25"
age_int = int(age_str)
print(f"Converted '{age_str}' to {age_int} of type {type(age_int)}")

# Integer to string
num = 42
num_str = str(num)
print(f"Converted {num} to '{num_str}' of type {type(num_str)}")

# String to float
price_str = "19.99"
price_float = float(price_str)
print(f"Converted '{price_str}' to {price_float} of type {type(price_float)}")

# List to set (removes duplicates)
numbers_list = [1, 2, 2, 3, 4, 4, 5]
numbers_set = set(numbers_list)
print(f"Converted {numbers_list} to {numbers_set} of type {type(numbers_set)}")
```

### 1.4 Checking Types

```python
value = 42

# Using type()
print(f"Type of value is: {type(value)}")
print(f"Is value an int? {type(value) is int}")

# Using isinstance()
print(f"Is value an int? {isinstance(value, int)}")
print(f"Is value a float? {isinstance(value, float)}")

# Check if value is a number (int or float)
print(f"Is value a number? {isinstance(value, (int, float))}")
```

---

## 2. Advanced Types <a name="advanced"></a>

### 2.1 Type Hints (Static Typing)

Python 3.5+ supports type hints, which allow you to specify the expected types of variables, function parameters, and return values. This helps with code documentation and can be used by tools like mypy for static type checking.

```python
from typing import List, Dict, Tuple, Set, Optional, Union, Any

# Function with type hints
def process_text(text: str, max_length: int = 100) -> str:
    """Process a text string and return a modified version."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

# Using the function
result = process_text("This is a sample text for processing.", 20)
print(result)
```

```python
# More complex type hints

# A function that takes a list of strings and returns a dictionary
def count_word_frequencies(words: List[str]) -> Dict[str, int]:
    """Count the frequency of each word in a list."""
    frequencies: Dict[str, int] = {}
    for word in words:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1
    return frequencies

# Using the function
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
word_counts = count_word_frequencies(words)
print(word_counts)
```

### 2.2 Custom Types with Classes

You can create custom types using classes. This is a fundamental concept in object-oriented programming.

```python
class TextDocument:
    def __init__(self, title: str, content: str, tags: List[str] = None):
        self.title = title
        self.content = content
        self.tags = tags or []

    def word_count(self) -> int:
        """Count the number of words in the document."""
        return len(self.content.split())

    def __str__(self) -> str:
        return f"TextDocument(title='{self.title}', words={self.word_count()}, tags={self.tags})"

# Create a document
doc = TextDocument(
    "Introduction to NLP",
    "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
    ["NLP", "AI", "introduction"]
)

print(doc)
print(f"Word count: {doc.word_count()}")
print(f"Type of doc: {type(doc)}")
```

```python
# Advanced type concepts using NamedTuple and TypedDict

from typing import NamedTuple, TypedDict

# NamedTuple - an immutable, typed version of a tuple with named fields
class Token(NamedTuple):
    text: str
    pos_tag: str
    is_stop: bool

# Create a token
token = Token(text="apple", pos_tag="NOUN", is_stop=False)
print(token)
print(f"Token text: {token.text}, POS tag: {token.pos_tag}")

# TypedDict - a dictionary with a fixed set of keys, each with a specified type
class DocumentMetadata(TypedDict):
    title: str
    author: str
    year: int
    keywords: List[str]

# Create document metadata
metadata: DocumentMetadata = {
    "title": "Advanced NLP Techniques",
    "author": "Jane Smith",
    "year": 2023,
    "keywords": ["NLP", "machine learning", "transformers"]
}
print(metadata)
```

```python
# Function types with Callable
from typing import Callable

# Define a type for text processing functions
TextProcessor = Callable[[str], str]

def apply_processors(text: str, processors: List[TextProcessor]) -> str:
    """Apply a series of text processors to a string."""
    result = text
    for processor in processors:
        result = processor(result)
    return result

# Define some text processors
def lowercase(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    import string
    return text.translate(str.maketrans("", "", string.punctuation))

# Apply processors
text = "Hello, World! This is a TEST."
processed_text = apply_processors(text, [lowercase, remove_punctuation])
print(f"Original: {text}")
print(f"Processed: {processed_text}")
```

### 2.3 Union Types and Optional

Union types allow a variable to have one of several types. Optional is a shorthand for Union[T, None].

```python
from typing import Union, Optional

# A function that can take either a string or a list of strings
def normalize_text(text: Union[str, List[str]]) -> str:
    if isinstance(text, list):
        return " ".join(text).lower()
    return text.lower()

# Examples
print(normalize_text("Hello World"))
print(normalize_text(["Hello", "World"]))

# Optional parameter (can be None)
def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> List[str]:
    """Extract entities of specified types from text."""
    # In a real implementation, this would use an NLP library
    # For demonstration, we'll just return some dummy data
    if entity_types is None:
        return ["John", "New York", "Google"]
    elif "PERSON" in entity_types:
        return ["John"]
    elif "LOCATION" in entity_types:
        return ["New York"]
    else:
        return []

# Examples
print(extract_entities("John works at Google in New York."))
print(extract_entities("John works at Google in New York.", ["PERSON"]))
```

---

## 3. Exercises <a name="exercises"></a>

Now it's your turn to practice working with Python types. Complete the following exercises to reinforce your understanding.

### Exercise 1: Type Conversion

Write a function `parse_numeric_data` that takes a list of strings, converts each string to a number (float or int as appropriate), and returns a list of numbers. If a string cannot be converted, it should be skipped.

```python
def parse_numeric_data(string_list: List[str]) -> List[Union[int, float]]:
    # Your code here
    pass

# Test cases
test_data = ["42", "3.14", "not a number", "99", "0.5"]
# Expected output: [42, 3.14, 99, 0.5]
```

### Exercise 2: Working with Dictionaries

Create a function `word_statistics` that takes a text string and returns a dictionary with the following statistics:
- 'word_count': the number of words in the text
- 'char_count': the number of characters (excluding spaces)
- 'avg_word_length': the average length of words
- 'unique_words': the number of unique words (case-insensitive)

```python
def word_statistics(text: str) -> Dict[str, Union[int, float]]:
    # Your code here
    pass

# Test case
sample_text = "Natural Language Processing is fascinating. NLP combines linguistics and computer science."
# Expected output: a dictionary with word_count, char_count, avg_word_length, and unique_words
```

### Exercise 3: Custom Types for NLP

Create a `Sentence` class that represents a sentence in an NLP context. It should:
- Store the original text
- Have a method to tokenize the sentence into words
- Have a method to count the frequency of each word
- Have a method to identify potential named entities (words that start with a capital letter, excluding the first word)

```python
class Sentence:
    def __init__(self, text: str):
        # Your code here
        pass

    def tokenize(self) -> List[str]:
        # Your code here
        pass

    def word_frequencies(self) -> Dict[str, int]:
        # Your code here
        pass

    def potential_named_entities(self) -> List[str]:
        # Your code here
        pass

# Test case
test_sentence = Sentence("John visited New York last summer with his friend Mary.")
# Expected: methods should return appropriate results for this sentence
```

### Exercise 4: Type Hints in Practice

Implement a function `process_documents` that takes a list of documents (each represented as a dictionary with 'id', 'title', and 'content' keys) and a processing function. The function should apply the processing function to each document's content and return a new list of documents with the processed content.

```python
from typing import List, Dict, Callable, Any

Document = Dict[str, Any]  # Type alias for a document
TextProcessor = Callable[[str], str]  # Type alias for a text processing function

def process_documents(documents: List[Document], processor: TextProcessor) -> List[Document]:
    # Your code here
    pass

# Test case
docs = [
    {"id": 1, "title": "Introduction", "content": "This is an introduction to NLP."},
    {"id": 2, "title": "Methods", "content": "We use various methods in NLP."}
]

def uppercase_processor(text: str) -> str:
    return text.upper()

# Expected: a new list of documents with uppercase content
```

---

## 4. Real-World Applications <a name="applications"></a>

Here are some examples of how Python's type system is used in real-world NLP projects:

### spaCy

spaCy is a popular NLP library that makes extensive use of Python's type system. It uses type hints throughout its codebase to ensure type safety and provide better documentation. The library defines custom types for tokens, documents, and other NLP concepts.

```python
import spacy
from spacy.tokens import Doc, Token, Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# 'doc' is of type Doc
# Each token in doc is of type Token
# Spans of tokens are of type Span
```

### Hugging Face Transformers

The Hugging Face Transformers library uses type hints to provide clear interfaces for its models and tokenizers. This helps users understand what types of inputs and outputs to expect when working with complex transformer models.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

# 'inputs' is a dictionary of torch.Tensor objects
# 'outputs' contains logits of type torch.Tensor
```

### NLTK

The Natural Language Toolkit (NLTK) is one of the oldest and most comprehensive NLP libraries. While it predates Python's type hints, it uses Python's object-oriented features to define clear types for linguistic concepts.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "Natural language processing is a field of computer science and linguistics."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

# 'tokens' is a list of strings
# 'tagged' is a list of tuples (word, tag)
```

### FastAPI

FastAPI is a modern web framework that leverages Python's type hints to automatically generate API documentation and perform request validation. It's often used to deploy NLP models as web services.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: TextRequest):
    # In a real application, this would use an NLP model
    return {"sentiment": "positive", "confidence": 0.95}
```

These examples demonstrate how Python's type system is used in real-world NLP applications to create more robust, maintainable, and well-documented code.
