import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data1 = pd.read_csv('votes_1975_2015.csv')

count_data = data1.groupby(['Country', 'Source Country']).size().reset_index(name='count')

# Filter out pairs with fewer than 3 instances
filtered_data = count_data[count_data['count'] >= 3]

# Merge the filtered data with the original data to retain only the valid pairs
merged_data = pd.merge(data1, filtered_data[['Country', 'Source Country']], on=['Country', 'Source Country'], how='inner')

# Remove rows where Source Country == Country
merged_data = merged_data[merged_data['Country'] != merged_data['Source Country']]

# Group the filtered data by 'Country' and 'Source Country', and calculate the mean of 'total_points'
data = merged_data.groupby(['Country', 'Source Country'])['total_points'].mean().reset_index()

# Initialize a dictionary to store quality for each country
# Initialize a list to store the biases
bias_list = []

# Calculate quality and bias for each (Country, Source Country) pair
for index, row in data.iterrows():
    ci = row['Source Country']
    cj = row['Country']
    vote_cij = row['total_points']
    
    # Exclude votes from both ci and cj
    relevant_points = data[(data['Country'] == cj) & (data['Source Country'] != ci) & (data['Source Country'] != cj)]
    # Calculate the mean of these points
    quality_cij = relevant_points['total_points'].mean()
    
    # Bias C_{ij} = Vote C_{ij} − Quality C_{ij}
    bias_cij = vote_cij - quality_cij
    
    # Append the result to the bias_list
    bias_list.append({'Source Country': ci, 'Country': cj, 'Bias': bias_cij})

# Convert the bias_list to a DataFrame
bias_df = pd.DataFrame(bias_list)

# Sort the biases in descending order
sorted_bias_df = bias_df.sort_values(by='Bias', ascending=False)
print(sorted_bias_df.head(10))

# Calculate the absolute value of each bias
bias_df['Abs_Bias'] = bias_df['Bias'].abs()

# Group by 'Source Country' and calculate the median absolute bias for each country
median_abs_bias = bias_df.groupby('Source Country')['Abs_Bias'].mean().reset_index()

# Sort the countries by the median absolute bias
least_biased_countries = median_abs_bias.sort_values(by='Abs_Bias')

# Print the top 10 least biased countries
print("Top 10 least biased countries:")
print(least_biased_countries.head(10))

# Optionally, you can plot the results using horizontal bars for better readability
plt.figure(figsize=(8, 7))
plt.barh(least_biased_countries['Source Country'], least_biased_countries['Abs_Bias'], color='skyblue')
plt.title('Mean Absolute Bias of Countries')
plt.xlabel('Mean Absolute Bias')
plt.ylabel('Country')
plt.gca().invert_yaxis()  # Invert y-axis to have the smallest bias at the top
plt.grid(True)
plt.tight_layout()
plt.show()