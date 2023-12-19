import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut


def generate_m(movies_idx, users, ratings):
    # Complete the datastructure for rating matrix 

    #All matrix values with -1.0
    m = {userId: {movieId: -1.0 for movieId in movies_idx} for userId in users}
    # m = {movieId: {userId: -1.0 for userId in users} for movieId in movies_idx}

    # Set matrix values efficiently
    for _, row in ratings.iterrows():
        m[row['userId']][row['movieId']] = row['rating']
        # m[row['movieId']][row['userId']] = row['rating']

    return m

def get_avg(ratings):
    rated_ratings = [rate for rate in ratings if rate != -1.0]

    if rated_ratings:
        return sum(rated_ratings) / len(rated_ratings)
    else:
        return None


def user_based_recommender(target_user_idx, matrix):
    target_user = matrix[target_user_idx]
    recommendations = []
    # Compute the similarity between  the target user and each other user in the matrix. 
    # We recomend to store the results in a dataframe (userId and Similarity)
    data = {'userId': [], 'Similarity': []}
    for userId, ratings in matrix.items(): 
        if userId != target_user_idx:  
            data['userId'].append(userId)
            data['Similarity'].append(sim.compute_similarity(list(target_user.values()), list(ratings.values())))

    df = pd.DataFrame(data)

    # n_most_similar = 10
    # simiar_users = df.nlargest(n_most_similar, 'Similarity')
    similar_users = df[df['Similarity'] > 0.95]

    # Determine the unseen movies by the target user. Those films are identfied 
    # since don't have any rating. 
     
    # Generate recommendations for unrated movies based on user similarity and ratings.
    # @ TODO 
    avg_rating_target = get_avg(list(target_user.values()))
    for movieId, rating in target_user.items(): 
        if rating == -1.0:
            expected_rate = avg_rating_target
            for _, row in similar_users.iterrows():
                similar_rate = matrix[row['userId']][movieId]
                if similar_rate != -1.0:
                    expected_rate += row['Similarity']*(similar_rate - get_avg(list(matrix[row['userId']].values())))
            recommendations.append((movieId, expected_rate))
    
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations

def normalize(w):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    magnitude = np.linalg.norm(w)
    tw_normalized = [weight/ magnitude for weight in w]

    return tw_normalized

def sim_vec(vec1, vec2):
    sim = 0
    for i in range(len(vec1)):
        sim += vec1[i]*vec2[i]
    return sim


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
    
    top5AnswerTrain = [recomendation[0] for recomendation in recommendations[:5]]

    userBasedRecommendationGenres = [0.0] * len(matrixmpa_genres.iloc[0])
    userBasedRecommendationGenres = np.array(userBasedRecommendationGenres)
    for recommendedMoviedId in top5AnswerTrain:
        userBasedRecommendationGenres += np.array(list(matrixmpa_genres.loc[recommendedMoviedId]))

    m_validate = generate_m(movies_idx, users_idy, ratings_val)
    target_user = m_validate[target_user_idx]
    sorted_movies_validate_dictionary = sorted(target_user.items(), key=lambda x: x[1], reverse=True)

    top5AnswerValidate = [item[0] for item in sorted_movies_validate_dictionary[:5]]

    userBasedValidationGenres = [0.0] * len(matrixmpa_genres.iloc[0])
    userBasedValidationGenres = np.array(userBasedValidationGenres)
    for recommendedMoviedId in top5AnswerValidate:
        userBasedValidationGenres += np.array(list(matrixmpa_genres.loc[recommendedMoviedId]))

    sim_user = sim_vec(normalize(userBasedRecommendationGenres), normalize(userBasedValidationGenres))

    print("Similitude between train and validation in user method: {}".format(sim_user))

    




