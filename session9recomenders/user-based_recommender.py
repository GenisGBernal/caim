import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut


def generate_m(movies_idx, users, ratings):
    # Complete the datastructure for rating matrix 

    x = ratings.iloc[2]

    #All matrix values with -1.0
    m = {userId: {movieId: -1.0 for movieId in movies_idx} for userId in users}

    # Set matrix values efficiently
    for _, row in ratings.iterrows():
        m[row['userId']][row['movieId']] = row['rating']

    return m 

def get_avg(ratings):
    rated_ratings = [rate for rate in ratings if rate != -1.0]

    if rated_ratings:
        return sum(rated_ratings) / len(rated_ratings)
    else:
        return None


def user_based_recommender(target_user_idx, matrix):
    # target_user = matrix.iloc[target_user_idx]
    target_user = matrix[target_user_idx]
    target_user_ratings = list(target_user.values())
    recommendations = []
    
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recomend to store the results in a dataframe (userId and Similarity)
    data = {'userId': [], 'Similarity': []}
    for userId, ratings in matrix.items():
        if userId != target_user_idx:
            rated_by_both_users_v1 = [v1 for v1, v2 in zip(target_user_ratings, ratings.values()) if v1 != -1.0 and v2 != -1.0]
            rated_by_both_users_v2 = [v2 for v1, v2 in zip(target_user_ratings, ratings.values()) if v1 != -1.0 and v2 != -1.0]
            data['userId'].append(userId)
            data['Similarity'].append(sim.compute_similarity(rated_by_both_users_v1, rated_by_both_users_v2))

    df = pd.DataFrame(data)

    # Determine the unseen movies by the target user. Those films are identfied 
    # since don't have any rating. 
    not_rated_movies = [movieId for movieId, rating in target_user.items() if rating == -1.0]
     
    # Generate recommendations for unrated movies based on user similarity and ratings.
    # @ TODO 
    avg_rating_target = get_avg(target_user_ratings)
    expected_rates = []
    for movieId in not_rated_movies:
        x = 0
        for index, row in df.iterrows():
            x += row['Similarity']*(matrix[row['userId']][movieId]- get_avg(matrix[row['userId']].values()))
        expected_rate = avg_rating_target + x
        expected_rates.append(expected_rate)
    
    return recommendations



if __name__ == "__main__":
    
    # Load the dataset
    path_to_ml_latest_small = 'C:/Users/thema/Desktop/proyectos/caim/lab/session9recomenders/ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)
        
    # user-to-user similarity
    target_user_idx = 123
    recommendations = user_based_recommender(target_user_idx, m)
     
    # The following code print the top 5 recommended films to the user
    for recomendation in recommendations[:5]:
        rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
        print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

    
    # Validation
    matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])
    
     
    








