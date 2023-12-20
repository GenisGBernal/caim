import pandas as pd
import numpy as np

import similarity as sim
import naive_recommender as nav
import utils as ut
import matplotlib.pyplot as plt


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


    similar_users_mean_rate = []
    for _, row in similar_users.iterrows():
        similar_users_mean_rate.append(get_avg(list(matrix[row['userId']].values())))
    similar_users['mean'] = similar_users_mean_rate
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
                    expected_rate += row['Similarity']*(similar_rate - row['mean'])
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
    path_to_ml_latest_small = 'D:/Users/M/Documents/Universidad/Quatrimestre_9/CAIM/caim/session9recomenders/ml-latest-small/'
    dataset = ut.load_dataset_from_source(path_to_ml_latest_small)

    # Ratings data
    val_movies = 5
    ratings_train, ratings_val = ut.split_users(dataset["ratings.csv"], val_movies)

    # Create matrix between user and movies 
    movies_idx = dataset["movies.csv"]["movieId"]
    users_idy = list(set(ratings_train["userId"].values))
    m = generate_m(movies_idx, users_idy, ratings_train)

    all_users = dataset["ratings.csv"]["userId"].unique()
    all_users = np.random.choice(all_users, 2, replace=False)
    # Inicializa listas para almacenar las similitudes de ambos métodos
    sim_naive_list = []
    sim_user_list = []
    users_list = []

    # Itera sobre todos los usuarios
    for target_user_idx in all_users:
        recommendations = user_based_recommender(target_user_idx, m)
        # user-to-user similarity
     
        # The following code print the top 5 recommended films to the user
        for recomendation in recommendations[:5]:
            rec_movie = dataset["movies.csv"][dataset["movies.csv"]["movieId"]  == recomendation[0]]
            print (" Recomendation :Movie:{} (Genre: {})".format(rec_movie["title"].values[0], rec_movie["genres"].values[0]))

        
        # Validation
        matrixmpa_genres = ut.matrix_genres(dataset["movies.csv"])

        #------------------------------ VALIDATION DATA ----------------------------------------------

        m_validate = generate_m(movies_idx, users_idy, ratings_val)
        target_user = m_validate[target_user_idx]
        sorted_movies_validate_dictionary = sorted(target_user.items(), key=lambda x: x[1], reverse=True)

        top5MoviesValidate = [item[0] for item in sorted_movies_validate_dictionary[:5]]

        validationGenres = [0.0] * len(matrixmpa_genres.iloc[0])
        validationGenres = np.array(validationGenres)
        for recommendedMoviedId in top5MoviesValidate:
            validationGenres += np.array(list(matrixmpa_genres.loc[recommendedMoviedId]))
        
        #------------------------------ USER BASED VALIDATION ----------------------------------------------
        top5AnswerTrain = [recomendation[0] for recomendation in recommendations[:5]]

        userBasedRecommendationGenres = [0.0] * len(matrixmpa_genres.iloc[0])
        userBasedRecommendationGenres = np.array(userBasedRecommendationGenres)
        for recommendedMoviedId in top5AnswerTrain:
            userBasedRecommendationGenres += np.array(list(matrixmpa_genres.loc[recommendedMoviedId]))

        sim_user = sim_vec(normalize(userBasedRecommendationGenres), normalize(validationGenres))

        print("Similitude between train and validation in user method: {}".format(sim_user))


        #------------------------------ NAIVE BASED VALIDATION ----------------------------------------------
        top5AnswerNaive = list(nav.naive_recommender(dataset["ratings.csv"], dataset["movies.csv"], 5)["movieId"])

        naiveValidationGenres = [0.0] * len(matrixmpa_genres.iloc[0])
        naiveValidationGenres = np.array(naiveValidationGenres)
        for recommendedMoviedId in top5AnswerNaive:
            naiveValidationGenres += np.array(list(matrixmpa_genres.loc[recommendedMoviedId]))

        sim_naive = sim_vec(normalize(naiveValidationGenres), normalize(validationGenres))

        print("Similitude between train and validation in naive method: {}".format(sim_naive))

        sim_naive_list.append(sim_vec(normalize(naiveValidationGenres), normalize(validationGenres)))

        # Calcula la similitud para el método User-Based
        sim_user_list.append(sim_vec(normalize(userBasedRecommendationGenres), normalize(validationGenres)))
        users_list.append(target_user_idx)
    # Calcula la similitud promedio para ambos métodos
    avg_sim_naive = np.mean(sim_naive_list)
    avg_sim_user = np.mean(sim_user_list)

    # Muestra los resultados
    
    print("Similitud promedio entre train y validation en método User-Based: {}".format(avg_sim_user))
    print("Similitud promedio entre train y validation en método Naive: {}".format(avg_sim_naive))

    users_list = [str(elemento) for elemento in users_list]
    iterations = list(range(1, len(sim_naive_list) + 1))

    # Crea un gráfico de líneas para mostrar las similitudes
    plt.plot(users_list, sim_naive_list, label='Recomanador Naive', marker='o')
    plt.plot(users_list, sim_user_list, label='Recomanador User-Based', marker='o')

    # Ajusta el formato del eje x para que solo muestre números enteros
    plt.xticks(rotation=45, ha='right', fontsize=8)

    # Añade etiquetas y título al gráfico
    plt.xlabel('Índex dels usuaris')
    plt.ylabel('Similaritat')
    plt.title('Similaritats entre el recomanador Naive i el User-based')
    plt.legend()  # Añade la leyenda para diferenciar las líneas

    # Agrega cuadrícula y estilo
    plt.grid(False)
    plt.style.use('default')

    # Muestra el gráfico
    plt.show()

