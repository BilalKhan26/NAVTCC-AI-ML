# even_number = [x for x in range(10) if x%2 == 0]

# numbers = iter([1,2,3,4])
# print(next(numbers))
# print(next(numbers))
# print(next(numbers))
# print(next(numbers))

# #Custom iterators
# class Counter:
#     def __init__(self, start, end):
#         self.current = start
#         self.end = end
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.current >= self.end:  # Stop when current reaches or exceeds end
#             raise StopIteration
#         value = self.current
#         self.current += 1
#         return value

# # Example usage:
# # if __name__ == "__main__":
#     counter = Counter(1, 5)
#     for num in counter:
#         print(num)  # Will print: 1, 2, 3, 4


# square = {x**2 for x in range(10)}
# # Result: {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

add = lambda x,y  : x + y
print(add(3,4))
    
