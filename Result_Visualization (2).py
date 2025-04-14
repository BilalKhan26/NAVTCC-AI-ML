import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_excel("6th class result 25.xlsx", skiprows=2)  # Skip header rows

# # 3. Histogram of Percentages
# percentages = df['Percentage'].dropna()

# # Create histogram
# plt.figure(figsize=(10, 6))
# plt.hist(percentages, bins=10, color='skyblue', edgecolor='black')
# plt.title('Distribution of Student Percentages (6th Class 2025)')
# plt.xlabel('Percentage (%)')
# plt.ylabel('Number of Students')
# plt.axvline(percentages.mean(), color='red', linestyle='dashed', label=f'Mean: {percentages.mean():.2f}%')
# plt.legend()
# plt.show()

# # 4. Bar Chart for Top 3 Performers
# top_3 = df.nlargest(3, 'Percentage')[['Name of Student', 'Percentage']]

# # Create bar chart
# plt.figure(figsize=(8, 5))
# plt.bar(top_3['Name of Student'], top_3['Percentage'], color='green', edgecolor='black')
# plt.title('Top 3 Performers (6th Class 2025)')
# plt.xlabel('Student Name')
# plt.ylabel('Percentage (%)')
# for i, v in enumerate(top_3['Percentage']):
#     plt.text(i, v + 1, f'{v:.2f}%', ha='center')
# plt.show()

# # 5. Box Plot for Subject-wise Performance
# subjects = ['Urdu', 'English', 'Maths', 'Science', 'Islamiat', 'Nazra-Quran', 
#             'Socail Studies', 'Al-Quran', 'Practical', 'Home Economics']
# subject_data = df[subjects].dropna()

# # Create box plot
# plt.figure(figsize=(12, 6))
# plt.boxplot(subject_data.values, labels=subjects, patch_artist=True, showfliers=False)
# plt.title('Subject-wise Performance Distribution (6th Class 2025)')
# plt.xlabel('Subjects')
# plt.ylabel('Scores')
# plt.xticks(rotation=45)
# plt.show()



# Creating a Pandas DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
print("Pandas DataFrame:")
print(df)
df.replace(6, 60, inplace=True)
print("DataFrame after replacing value 6 with 60:")
print(df)
# Index Setting
df.set_index('A', inplace=True)
print("DataFrame with new index:")
print(df)
print(df.iloc[1])#Interger-Location = iloc()
print(df.loc[1])#Label-Location = loc()
print(df.loc[1:2])#Label-Location = loc()
