#List
fruits = ["banana","pineapple","grape","apple"]
print(fruits[2])
fruits[1] = "cherry" #replaces pineapple with cherry at index 1
fruits.append("tomato") #places tomato at end of list
fruits.remove("banana") #remove the index where banana is placed
print(fruits)
print(fruits)
print(fruits)
#Tuple
fruits = ("banana","pineapple","grape","apple")
print(fruits.count(2))
print(f"Lenght of tuple",{len(fruits)})
#Dictionary
person = {
    "Name":"Ashbal",
    "Age": 22,
    "Qualification":"Grad"
}
print(person)
print(person["Name"]) #Display Key Value
person["Age"] = 35
print(person) #Change value of Key
person["Job"] = "Engineer" #Add new Unique Key
print(person)
del person ["Job"]
print(person)
#Set
numbers = {1,2,3,4,5,6}
numbers.add(7)
numbers.remove(3)
print(numbers)