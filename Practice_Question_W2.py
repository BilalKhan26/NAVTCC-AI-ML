class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.available = True
    
    def borrow(self):
        if self.available:
            self.available = False
            return f"{self.title} has been borrowed"
        return f"{self.title} is not available"
    
    def return_book(self):
        if not self.available:
            self.available = True
            return f"{self.title} has been returned"
        return f"{self.title} is already in library"

# Demonstration
book1 = Book("Python Basics", "John Doe")
book2 = Book("OOP Concepts", "Jane Smith")
print(book1.borrow())  # Python Basics has been borrowed
print(book2.borrow())  # OOP Concepts has been borrowed
print(book1.return_book())  # Python Basics has been returned
print(book2.borrow())  # OOP Concepts is not available


class Student:
    university = "ABC University"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @classmethod
    def update_university(cls, new_name):
        cls.university = new_name
        return f"University updated to {new_name}"

# Demonstration
student1 = Student("Alice", 20)
student2 = Student("Bob", 22)
print(f"{student1.name}: {student1.university}")  # Alice: ABC University
print(f"{student2.name}: {student2.university}")  # Bob: ABC University
Student.update_university("XYZ University")
print(f"{student1.name}: {student1.university}")  # Alice: XYZ University
print(f"{student2.name}: {student2.university}")  # Bob: XYZ University


class Employee:
    employee_count = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1
    
    @classmethod
    def total_employees(cls):
        return f"Total employees: {cls.employee_count}"

# Demonstration
emp1 = Employee("Alice", 50000)
emp2 = Employee("Bob", 60000)
emp3 = Employee("Charlie", 55000)
print(Employee.total_employees())  # Total employees: 3
print(emp1.total_employees())      # Total employees: 3


class MathOperations:
    @staticmethod
    def add_numbers(a, b):
        return a + b

# Demonstration
print(MathOperations.add_numbers(5, 3))  # 8
math_obj = MathOperations()
print(math_obj.add_numbers(10, 20))      # 30


class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def display_info(self):
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    def __init__(self, brand, model, number_of_doors):
        super().__init__(brand, model)
        self.number_of_doors = number_of_doors
    
    def display_info(self):
        return f"{super().display_info()} with {self.number_of_doors} doors"

class SportsCar(Car):
    def __init__(self, brand, model, number_of_doors, top_speed):
        super().__init__(brand, model, number_of_doors)
        self.top_speed = top_speed
    
    def display_info(self):
        return f"{super().display_info()}, Top Speed: {self.top_speed} mph"

# Demonstration
sports_car = SportsCar("Ferrari", "488", 2, 205)
print(sports_car.display_info())  # Ferrari 488 with 2 doors, Top Speed: 205 mph


class BankAccount:
    def __init__(self):
        self.__balance = 0
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew ${amount}"
        return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        return f"Current balance: ${self.__balance}"

# Demonstration
account = BankAccount()
print(account.deposit(1000))      # Deposited $1000
print(account.withdraw(500))      # Withdrew $500
print(account.withdraw(600))      # Insufficient funds or invalid amount
print(account.get_balance())      # Current balance: $500


def logger():
    message_count = 0
    
    def log_message(message):
        nonlocal message_count
        message_count += 1
        return f"Message {message_count}: {message}"
    
    return log_message

# Demonstration
log = logger()
print(log("First message"))   # Message 1: First message
print(log("Second message"))  # Message 2: Second message
print(log("Third message"))   # Message 3: Third message


import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(2)
    return "Done"

# Demonstration
print(slow_function())  # Done
                        # slow_function took 2.00 seconds


def generate_numbers(n):
    for i in range(1, n + 1):
        yield i

# Demonstration
for num in generate_numbers(5):
    print(num, end=" ")  # 1 2 3 4 5


def read_file(filename):
    file = None
    try:
        file = open(filename, 'r')
        content = file.read()
        print(f"Content of {filename}:\n{content}")
    except FileNotFoundError:
        print(f"File {filename} not found")
    finally:
        if file is not None:
            file.close()
            print(f"File {filename} closed")

# Demonstration (assuming test.txt exists)
# read_file("test.txt")
read_file("nonexistent.txt")  # File nonexistent.txt not found
                              # File nonexistent.txt closed


import pickle

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
    
    def __str__(self):
        return f"User: {self.username}"

# Save user
user = User("john_doe", "secure123")
with open("user.pkl", "wb") as f:
    pickle.dump(user, f)

# Load user
with open("user.pkl", "rb") as f:
    loaded_user = pickle.load(f)
    print(loaded_user)          # User: john_doe
    print(loaded_user.password) # secure123



# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# def copy_image(original_path, copy_path):
#     # Read and write image in binary mode
#     with open(original_path, 'rb') as source:
#         with open(copy_path, 'wb') as target:
#             target.write(source.read())
    
#     # Display both images
#     original = mpimg.imread(original_path)
#     copy = mpimg.imread(copy_path)
    
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     plt.imshow(original)
#     plt.title("Original")
#     plt.subplot(122)
#     plt.imshow(copy)
#     plt.title("Copy")
#     plt.show()

# Demonstration (replace with actual image paths)
# copy_image("original.jpg", "copy.jpg")