import numpy as np
import random
import scipy


"""Demonstrates practical NumPy applications from Numpy.pdf"""

print("\n=== Practical NumPy Applications ===\n")
# 1. Hottest day on average
temperatures = np.random.randint(15, 40, size=(7, 24))
daily_means = np.mean(temperatures, axis=1)
hottest_day = np.argmax(daily_means)
print(f"1. Hottest day was day {hottest_day + 1} with average {daily_means[hottest_day]:.1f}°C")

# 2. Product with highest total sales
sales = np.random.randint(50, 200, size=(5, 6))
total_sales = np.sum(sales, axis=1)
best_product = np.argmax(total_sales)
print(f"\n2. Product {best_product + 1} had highest sales: {total_sales[best_product]} units")

# 3. Day with fewest hospital visits
visits = np.random.randint(50, 150, size=30)
quietest_day = np.argmin(visits)
print(f"\n3. Day {quietest_day + 1} had fewest visits: {visits[quietest_day]}")

# 4. Highest and lowest subject marks
marks = np.array([85, 92, 78, 88, 65, 90, 72, 80])
highest_subj = np.argmax(marks)
lowest_subj = np.argmin(marks)
print(f"\n4. Highest mark in subject {highest_subj + 1} ({marks[highest_subj]}), lowest in subject {lowest_subj + 1} ({marks[lowest_subj]})")

# 5. Average likes per week
daily_likes = np.random.randint(100, 500, size=28)
weekly_likes = daily_likes.reshape(4, 7)
weekly_avg = np.mean(weekly_likes, axis=1)
for week, avg in enumerate(weekly_avg, 1):
    print(f"\n5. Week {week} average: {avg:.1f} likes")

# 6. Increase salaries by 10%
salaries = np.array([45000, 52000, 38000, 61000])
new_salaries = salaries * 1.10
print("\n6. New salaries:", new_salaries)

# 7. Day with most rain
hourly_rain = np.random.uniform(0, 5, size=24*30)
daily_rain = hourly_rain.reshape(30, 24).sum(axis=1)
wettest_day = np.argmax(daily_rain)
print(f"\n7. Day {wettest_day + 1} had most rain: {daily_rain[wettest_day]:.1f}mm")

# 8. Combine cricket team scores
team_a = np.array([45, 78, 92, 65])
team_b = np.array([67, 54, 81, 73])
combined = np.concatenate([team_a, team_b])
print("\n8. Combined scores:", combined)

# 9. Student with highest average
scores = np.random.randint(50, 100, size=(5, 3))
averages = np.mean(scores, axis=1)
best_student = np.argmax(averages)
print(f"\n9. Student {best_student + 1} has highest average: {averages[best_student]:.1f}")

# 10. Split power usage into daily sections
minute_data = np.random.uniform(0.5, 5, size=96*30)
daily_data = minute_data.reshape(30, 96)
print("\n10. Daily data shape:", daily_data.shape)

# 11. Maximum temperature each day
hourly_temps = np.random.randint(20, 50, size=(30, 24))
daily_max = np.max(hourly_temps, axis=1)
print("\n11. Daily max temperatures:", daily_max)

# 12. Remove duplicate customer IDs
customer_ids = np.array([101, 102, 101, 103, 102, 104])
unique_ids = np.unique(customer_ids)
print("\n12. Unique customer IDs:", unique_ids)

# 13. Analyze gym member improvements
weekly_weights = np.random.uniform(50, 100, size=(10, 4))  # 10 members, 4 weeks
improvements = weekly_weights[:, -1] - weekly_weights[:, 0]
print("\n13. Member improvements over 4 weeks:", improvements)

# 14. Count transactions > $10,000
transactions = np.random.randint(100, 20000, size=1000)
large_transactions = np.sum(transactions > 10000)
print("\n14. Transactions > $10,000:", large_transactions)

# 15. Count 5-star ratings
ratings = np.random.randint(1, 6, size=100)
five_stars = np.sum(ratings == 5)
print("\n15. 5-star ratings:", five_stars)

# 16. Shortest and longest level durations
level_times = np.random.uniform(1, 30, size=20)
print("\n16. Shortest level:", np.min(level_times), "minutes")
print("Longest level:", np.max(level_times), "minutes")

# 17. Calculate attendance percentage
attendance = np.random.randint(0, 2, size=30)
attendance_pct = np.mean(attendance) * 100
print(f"\n17. Attendance percentage: {attendance_pct:.1f}%")

# 18. Find busiest week
# daily_visits = np.random.randint(50, 200, size=60)
# weekly_visits = daily_visits.reshape(7).sum(axis=1)  # 8 weeks (with some extra days)
# busiest_week = np.argmax(weekly_visits) + 1
# print(f"\n18. Busiest week was week {busiest_week} with {weekly_visits[busiest_week-1]} visits")

