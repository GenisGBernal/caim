import pandas as pd 
import utils as ut

def naive_recommender(ratings: object, movies:object, k: int = 10) -> list: 
    # Provide the code for the naive recommender here. This function should return 
    # the list of the top most viewed films according to the ranking (sorted in descending order).
    # Consider using the utility functions from the pandas library.
    most_seen_movies= []

    movies_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
    movies_stats.columns = ['movieId', 'rating_mean', 'rating_count']

    movies_stats_title = pd.merge(movies_stats, movies, on='movieId')

    m = 10
    C = ratings['rating'].mean()

    movies_stats_title['weighted_rating'] = ((movies_stats_title['rating_count'] * movies_stats_title['rating_mean'] + m * C) / (movies_stats_title['rating_count'] + m)).round(2)
    
    most_seen_movies = movies_stats_title.sort_values(by='weighted_rating', ascending=False)[['movieId', 'title', 'weighted_rating']]

    return most_seen_movies[:k]


if __name__ == "__main__":
    
    path_to_ml_latest_small = 'D:/Users/M/Documents/Universidad/Quatrimestre_9/CAIM/caim/session9recomenders/ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)
    
    ratings, movies = dataset["ratings.csv"], dataset["movies.csv"]
    naive_recommender(ratings, movies)

