# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Read the CSV file and assign it to a variable
data = pd.read_csv('imdb_top_1000.csv')
# Display the shape of the DataFrame
print(data.shape)
# Print all columns of the DataFrame
print(data.columns)
# Remove the 'Poster_Link' column from the DataFrame
data = data.drop('Poster_Link', axis=1)
# Verify the updated DataFrame
print(data.head())
# Set 'Series_Title' column as the index
data.set_index('Series_Title', inplace=True)
# Verify the updated DataFrame with the new index
print(data.head())
# Extract the 'Genre' column and split the genre values
genres = data['Genre'].str.split(',').explode().str.strip()
# Get the unique genres
unique_genres = genres.unique()
# Print the unique genres
print("Unique Genres:")
for genre in unique_genres:
    print(genre)
# Count the number of unique directors
num_directors = data['Director'].nunique()
# Print the number of directors
print("Number of Directors:", num_directors)
# Calculate the mean, median, and standard deviation of IMDB rating
imdb_rating_mean = data['IMDB_Rating'].mean()
imdb_rating_median = data['IMDB_Rating'].median()
imdb_rating_std = data['IMDB_Rating'].std()
# Print the mean, median, and standard deviation of IMDB rating
print("Mean IMDB Rating:", imdb_rating_mean)
print("Median IMDB Rating:", imdb_rating_median)
print("Standard Deviation of IMDB Rating:", imdb_rating_std)
# Count the number of movies directed by each director
director_counts = data['Director'].value_counts()
# Get the director with the highest count
most_movies_director = director_counts.idxmax()
# Print the director with the most number of movies
print("Director with the most number of movies:", most_movies_director)
# Convert the 'Runtime' column to numeric
data['Runtime'] = pd.to_numeric(data['Runtime'], errors='coerce')
# Filter out rows with missing genre values
# Find the index label with the highest runtime
max_runtime_index = data['Runtime'].idxmax()
# Get the movie title with the highest runtime
movie_highest_runtime = max_runtime_index
print("Movie with the highest runtime:", movie_highest_runtime)
# Convert 'Year' column to integer data type
data['Released_Year'] = pd.to_numeric(data['Released_Year'], errors='coerce')
# Filter the DataFrame for movies released after year 2000 and IMDB rating > 8.5
filtered_data = data[(data['Released_Year'] > 2000) & (data['IMDB_Rating'] > 8.5)]
# Count the number of movies in the filtered DataFrame
num_movies = len(filtered_data)
# Print the number of movies released after year 2000 with IMDB rating > 8.5
print("Number of movies released after year 2000 with IMDB rating > 8.5:", num_movies)
# Count the number of movies directed by each director
director_counts = data['Director'].value_counts()
# Count the number of movies directed by each director
director_counts = data['Director'].value_counts()
# Create a countplot using seaborn
plt.figure(figsize=(12, 6))
sns.countplot(data=data, y='Director', order=director_counts.index)
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.title('Count of Movies Directed by Each Director')
# Adjust y-axis tick parameters
plt.tick_params(axis='y', which='major', labelsize=1)  # Set font size
plt.tight_layout()  # Adjust the spacing to prevent label overlap
plt.show()
# Create a histogram plot of IMDB ratings
plt.figure(figsize=(8, 6))
plt.hist(data['IMDB_Rating'], bins=20, edgecolor='black')
plt.xlabel('IMDB Rating')
plt.ylabel('Frequency')
plt.title('Distribution of IMDB Ratings')
plt.show()
# Sort the DataFrame by Meta score in descending order
top_movies = data.sort_values(by='Meta_score', ascending=False).head(5)
# Get the movie titles of the top 5 movies
top_movies_titles = top_movies.index
print("Top 5 movies with the highest Meta scores:")
print(top_movies_titles)






      