# 3. Product Review Ratings
# Scenario: You are analyzing review ratings (1 to 5 stars) for a product on an e-commerce platform.
# Question: Use a bar plot or histogram to understand the distribution of ratings. What does the plot tell you about customer satisfaction?

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Generate sample data (replace this with your actual data)
num_ratings = 100
ratings = np.random.randint(1, 6, num_ratings)
spendings  = 1000
# monthly = np.random.randint(100,50000,spendings)
# print(monthly)

# Create a bar plot
plt.figure(figsize=(8, 6))
# plt.bar(range(1, 6), np.bincount(ratings), color='skyblue')
plt.subplot(1,2,1)
# plt.hist(ratings,bins =20, color='skyblue',edgecolor ="black")
sns.histplot(ratings,bins =20,kde = True, color='skyblue',edgecolor ="black")
# sns.barplot(x = ratings, color = 'red', edgecolor = 'black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Product Review Ratings')
plt.xticks(range(1, 6))
plt.grid(axis='y')
# Show the plot
plt.show()