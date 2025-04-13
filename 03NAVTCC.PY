age = 20
if age < 18:
    print("Person is Minor.")
elif age == 18:
    print("Your are just an adult.")
else:
    print("You're an Adult ")
# -----------------------------------------------
age = 25
city = "New York"
if age >= 18:
    if city == "New York":
        print("You're adult in New York.")
    else:
        print("You're adult from another city.")
else:
    print("You're a minor")
# --------------------------------------------------
status = "Adult" if age >= 18 else "Minor"
print(status)
# ----------------------------------------------------
for i in range(10):
    if i == 5:
        break
    print(i)
# ----------------
for i in range(10):
    if i == 5:
        continue # skip if-teration and excludes 5
    print(i)
# --------------
for i in range(10):
    if i == 5:
        pass # skip if-iteration and includes 5
    print(i)
#-----------------------------------
