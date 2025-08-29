import numpy as np 
import pandas as pd
from collections import defaultdict

data_path = 'Movie Data/ml-latest-small/ratings.csv'
read_data = pd.read_csv("Movie Data/ml-latest-small/ratings.csv")

#number of unique users
n_users = read_data["userId"].nunique()

#number of unique movies
n_movies = read_data["movieId"].nunique()

#loading rating data from csv file
def load_rating_data_csv(data_path, n_users, n_movies):

    #creating user * movie rating matrix filled with zeros. Rows= users, columns= movies
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {} #dictionary to map real movieID 
    movie_n_rating = defaultdict(int) #keeping count how many ratings each movie has received keys = movieId, Values = count of rating
    with open(data_path, 'r') as file: # openg the rating csv file in read mode
        next(file) #skiping the header line of csv file
        for line in file:
            user_id, movie_id, rating, _ = line.strip().split(",") #strippping all column by "," with user_id, movie_id and rating
            user_id = int(user_id) -1 #Convert to integer, and subtract 1 because user IDs start at 1 in the dataset but numpy arrays are 0-indexed.
            if movie_id not in movie_id_mapping: 
                movie_id_mapping[movie_id] = len(movie_id_mapping) #if moive_id in new than shifting that movie_id to next available column index. 
            rating = float(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating #Convert rating to float and store it in the matrix at position [user, movie_index]
            if rating > 0:
                movie_n_rating[movie_id] +=1 #calculating every movie's total rating 
    return data, movie_n_rating, movie_id_mapping 


data, movie_n_rating, movie_id_mapping =  load_rating_data_csv(data_path, n_users, n_movies)

#checking data distribution: from 1-5 ratting how many movie is ratted
def display_distribution(data): 
    values, counts = np.unique(data, return_counts=True)  #finds all unique values in data amd return_counts returns how many times unique value is appeared
    for value, count in zip(values, counts): 
        print(f'Number of rating {value}: {count}')
display_distribution(data)

#finding most rated movie
movie_rating_lists = list(movie_n_rating.items()) #converting dictionary into lists of movie_id, num_ratings
#sort by number of ratings (second in the tuple) in descending order
sorted_movies = sorted(movie_rating_lists, key=lambda d:d[1], reverse=True)
#picking the top one rated movie
movie_id_most, n_rating_most = sorted_movies[0]
print(f"MovieID {movie_id_most} has highest {n_rating_most} rating.")

#since highest rated movie is found we have to target that movie and organized the data accordingly
X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1) #delete the entire column of that higest rated movie from data. Now all users rating are available except the highest one
Y_raw = data[:, movie_id_mapping[movie_id_most]] #select the column of higest rated movie for all users

#Y_raw>0 creates a boolean mask (True if user rated the movie, False if not).Y_raw = [5, 0, 3, 0, 4] Y_raw > 0 → [True, False, True, False, True]
  
X = X_raw[Y_raw >0] # filter only those user's rating who have rated the highest rated movie
Y = Y_raw[Y_raw>0] #actual ratings of that highest rated movie. 
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

display_distribution(Y)