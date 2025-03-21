# Python Classes

This notebook explores Python classes, which are crucial for object-oriented programming (OOP). Mastering classes helps structure code effectively—especially important in large NLP or data science projects.

[Raw Notebook](https://github.com/agombert/AdvancedNLPClasses/blob/main/notebooks/support/Session_1_1_Python_1o1_2.ipynb)

## Table of Contents
1. [Basic Concepts](#basic)
2. [Advanced Concepts](#advanced)
3. [Exercises](#exercises)
4. [Real-World Applications](#applications)

---

## 1. Basic Concepts <a name="basic"></a>

In Python, classes allow you to create custom data types and bundle data (attributes) with behaviors (methods).

### 1.1 Defining a Class

```python
# Define a simple Animal class with an initializer and a method.
class Animal:
    def __init__(self, name, species):
        self.name = name      # instance attribute for the animal's name
        self.species = species  # instance attribute for the animal's species

    def make_sound(self):
        # Generic animal sound method, to be overridden by subclasses
        print("<generic animal sound>")

# Instantiate the class with a dog's information
dog = Animal("Fido", "Canine")
print(dog.name, dog.species)  # Should print: Fido Canine
dog.make_sound()             # Calls the generic make_sound() method
```

### 1.2 Inheritance

Inheritance enables a new class (child) to inherit attributes and methods from an existing class (parent).

```python
# Define a Dog class that inherits from Animal.
class Dog(Animal):
    def __init__(self, name):
        # Call the parent class __init__ method with species preset as "Canine"
        super().__init__(name, "Canine")

    def make_sound(self):
        # Override make_sound() to print a dog-specific sound.
        print("Woof!")

# Instantiate a Dog object and demonstrate inherited behavior
my_dog = Dog("Rex")
print(my_dog.name, my_dog.species)  # Should print: Rex Canine
my_dog.make_sound()               # Should print: Woof!
```

---

## 2. Advanced Concepts <a name="advanced"></a>

### 2.1 Class Methods and Static Methods

Class methods take a reference to the class (`cls`) instead of the instance (`self`), while static methods don’t take any special first argument.

```python
class MathUtils:
    PI = 3.14159  # Class attribute shared among all instances

    @classmethod
    def circle_area(cls, radius):
        # Calculates area using the class attribute PI
        return cls.PI * (radius ** 2)

    @staticmethod
    def add(a, b):
        # Simply returns the sum of a and b (no reference to the class)
        return a + b

# Using class and static methods without creating an instance
print(MathUtils.circle_area(5))  # Computes area of a circle with radius 5
print(MathUtils.add(10, 20))     # Computes the sum of 10 and 20
```

### 2.2 Dunder Methods (Magic Methods)

Dunder (double underscore) methods allow classes to integrate with Python’s built-in operations (like `str()`, `len()`, arithmetic, etc.).

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        # This dunder method defines how to print the Vector instance.
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        # Overloads the + operator to add two vectors component-wise.
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(2, 3)
v2 = Vector(4, 1)
v3 = v1 + v2  # Uses the overloaded __add__ method
print(v3)    # Should print: Vector(6, 4)
```

### 2.3 Composition vs Inheritance

Composition is an alternative to inheritance. Instead of inheriting from a class, you hold an instance of another class as an attribute.

```python
class Engine:
    def start(self):
        print("Engine starts.")
    def stop(self):
        print("Engine stops.")

class Car:
    def __init__(self):
        # Composition: Car has an Engine instance as an attribute.
        self.engine = Engine()

    def drive(self):
        self.engine.start()  # Use the engine's start method
        print("Car is driving...")
        self.engine.stop()   # Use the engine's stop method

my_car = Car()
my_car.drive()  # Demonstrates the use of composition in Car class
```

---

## 3. Exercises <a name="exercises"></a>

Work on the following exercises to consolidate your understanding of Python classes.

### Exercise 1: Creating a Book Class
1. Create a `Book` class with attributes: `title`, `author`, `pages`.
2. Implement a `__str__` method that returns a string in the format: `"Book(title='...', author='...', pages=...)"`.
3. Instantiate a few `Book` objects and print them.

```python
# Your solution here
class Book:
    def __init__(self, title: str, author: str, pages: int):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self) -> str:
        # Returns a formatted string representation of the book.
        return f"Book(title='{self.title}', author='{self.author}', pages={self.pages})"

# Test your class:
b1 = Book("Brave New World", "Aldous Huxley", 311)
print(b1)
b2 = Book("1984", "George Orwell", 328)
print(b2)
```

### Exercise 2: Inheritance
1. Create a `Vehicle` parent class with attributes: `make`, `model`, and a method `drive()`.
2. Create a `Truck` child class that inherits from `Vehicle`. Add an attribute `capacity` and override `drive()` to print a different message.
3. Instantiate both classes and call their `drive()` methods.

```python
# Your solution here
class Vehicle:
    def __init__(self, make: str, model: str):
        self.make = make
        self.model = model

    def drive(self):
        print(f"The {self.make} {self.model} is driving.")

class Truck(Vehicle):
    def __init__(self, make: str, model: str, capacity: int):
        # Initialize parent class attributes using super()
        super().__init__(make, model)
        self.capacity = capacity

    def drive(self):
        # Override drive() with a truck-specific message.
        print(f"The {self.make} {self.model} truck with capacity {self.capacity}kg is hauling.")

# Example usage:
car = Vehicle("Toyota", "Camry")
car.drive()

pickup = Truck("Ford", "F-150", 1000)
pickup.drive()
```

### Exercise 3: Class/Static Methods
1. Define a class `MathOperations` with a class method `from_list(values)` that returns an instance with some aggregated result (e.g., sum of the list), and a static method `multiply(a, b)`.
2. Demonstrate usage by creating an instance using `from_list` and calling `multiply`.

```python
# Your code here
class MathOperations:
    def __init__(self, total: float):
        self.total = total

    @classmethod
    def from_list(cls, values: list) -> "MathOperations":
        # Aggregate the list values by summing them.
        total = sum(values)
        return cls(total)

    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b

# Example usage:
ops = MathOperations.from_list([1, 2, 3])
print(ops.total)  # Expected output: 6
print(MathOperations.multiply(3, 5))  # Expected output: 15
```

---

## 4. Real-World Applications <a name="applications"></a>

### Frameworks Using OOP
- **Django**: A popular web framework that relies heavily on classes (models, views) to structure large applications.
- **Flask Extensions**: Often define extension classes for plugin functionality.

### NLP Libraries
- **spaCy**: Defines classes like `Doc`, `Token`, `Span` for text processing.
- **NLTK**: Many classes for parsing, tokenization, etc.

### Data Science
- **scikit-learn**: Almost every algorithm is an object with `.fit()` and `.predict()` methods.
- **PyTorch / TensorFlow**: Model classes that define neural network architectures.

Classes are the foundation of these libraries, making it easier to organize and extend complex functionality.
