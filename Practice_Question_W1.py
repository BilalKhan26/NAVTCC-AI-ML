#Manage Score
# user_list = []
# n = int(input("How many items would you like to enter? "))

# def scoreboard(n):
#     # Input scores
#     for i in range(n):
#         # Convert input to float since we're dealing with scores
#         item = float(input(f"Enter score {i+1}: "))
#         user_list.append(item)

#     print("Score List:", user_list)
    
#     option = int(input("Which operation would you like to do: \n 1 for average, 2 for remove desired index,\n 3 for inserting value at any index,\n 4 to sort the list,\n 5 to find min and max value of list = \n "))
    
#     if option == 1:
#         average = sum(user_list) / n
#         print(f"Average score: {average}")
        
#     elif option == 2:
#         print("Current list:", user_list)
#         index = int(input("Enter index to remove (0 to {}): ".format(len(user_list)-1)))
#         if 0 <= index < len(user_list):
#             removed_item = user_list.pop(index)  # pop() returns the removed item
#             print(f"Removed {removed_item}. Updated list:", user_list)
#         else:
#             print("Invalid index!")
            
#     elif option == 3:
#         index = int(input("Enter index to insert at (0 to {}): ".format(len(user_list))))
#         value = float(input("Enter score to insert: "))
#         if 0 <= index <= len(user_list):
#             user_list.insert(index, value)
#             print("Updated list:", user_list)
#         else:
#             print("Invalid index!")
            
#     elif option == 4:
#         user_list.sort()
#         print("Sorted list:", user_list)
        
#     elif option == 5:
#         minimum = min(user_list)
#         maximum = max(user_list)
#         print(f"Minimum score: {minimum}")
#         print(f"Maximum score: {maximum}")
        
#     else:
#         print("Invalid option! Please choose 1-5")

# scoreboard(n)

#Online Store System
# Item_product = []
# flag = False
# while True:
#     product_name = input("Enter name: ")
#     price = int(input("Enter price: "))
#     category = input("Enter category: ")

#     Item_product.append({"product_name": product_name, "price": price, "category": category})

#     leave = int(input("Exit = enter 1, Continue = enter 0: "))
#     print(leave)
#     if leave == 1:
#         break

# print("All products:", Item_product)

# find = input("Which product needed: ")
# flag = False  # Already declared above, but good practice to reset it here
# for product in Item_product:
#     if product["product_name"] == find:
#         flag = True
#         print("Found product:", product)

# if not flag:
#     print(f"Not Found: {find}")  # Fixed formatting syntax and removed unnecessary curly braces


contacts = {"John": "123-456-7890", "Alice": "987-654-3210"}
contacts["Bob"] = "555-555-5555"

contacts["John"] = "111-222-3333"  # Update John's phone number
del contacts["Alice"]  # Remove Alice's contact
"Bob" in contacts  # Returns True if Bob is in the phonebook

emails = set()
emails.add("user@example.com")
"user@example.com" in emails  # Returns True if already signed up

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib_gen = fibonacci()
for _ in range(10):
    print(next(fib_gen))


def calculate_salary(hours_worked, hourly_rate):
    return hours_worked * hourly_rate

def calculate_salary_with_bonus(hours_worked, hourly_rate, bonus=0):
    return (hours_worked * hourly_rate) + bonus


def get_ticket_price(age):
    base_price = 10  # Base price for a ticket
    if age <= 12:  # Child discount
        return base_price * 0.5  # 50% discount
    elif age >= 65:  # Senior discount
        return base_price * 0.7  # 30% discount
    else:
        return base_price  # Full price for adults

print(get_ticket_price(5))   # Expected 5.0 (child discount)
print(get_ticket_price(30))  # Expected 10.0 (full price)
print(get_ticket_price(70))  # Expected 7.0 (senior discount)

students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
sorted_by_score = sorted(students, key=lambda student: student[1], reverse=True)

students = [("Alice", 85, 20), ("Bob", 92, 22), ("Charlie", 78, 19)]
sorted_by_age = sorted(students, key=lambda student: student[2])


def calculate_product(n):
    product = 1
    for i in range(1, n + 1):
        product *= i
    return product

print(calculate_product(5))  # 5! = 120
print(calculate_product(6))  # 6! = 720


prices = [100, 200, 300]
discount = 0.1  # 10% discount
discounted_prices = [price * (1 - discount) for price in prices]

expensive_items = [price for price in prices if price > 150]
total_earnings = sum(prices)
total_discounted_earnings = sum(discounted_prices)
