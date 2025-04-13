import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def build_utility_matrix(df):
    """
    Build the utility matrix.

    :param file_path: file path of ratings data
    :return: utility matrix
    """
    utility_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    return utility_matrix

def pearson_similarity(utility_matrix):
    """
    Calculate the Pearson similarity matrix.
    
    :param utility_matrix: utility matrix
    :return similarity_matrix: pandas.DataFrame, Pearson similarity matrix
    """
    similarity_matrix = utility_matrix.T.corr(method='pearson')
    return similarity_matrix

def adjusted_cosine_similarity(utility_matrix):
    """
    Calculate the adjusted cosine similarity matrix.
    
    :param utility_matrix: utility matrix
    :return similarity_matrix: pandas.DataFrame, adjusted cosine similarity matrix
    """
    users_mean = utility_matrix.mean(axis=1)
    adjusted_utility_matrix = utility_matrix.sub(users_mean, axis=0)
    # cosine_similarity expects no NaN values
    similarity_matrix = cosine_similarity(adjusted_utility_matrix.T.fillna(0))
    return pd.DataFrame(similarity_matrix, index=utility_matrix.columns, columns=utility_matrix.columns)

def predict_for_target_id(id, user_or_item, utility_matrix, similarity_matrix, k=200):
    """
    Predict the ratings for a item or for a user.
    User-User CF or Item-Item CF.

    :param user_id: user id or item id
    :param user_or_item: the id is for 'user' or 'item'
    :param utility_matrix: utility matrix
    :param similarity_matrix: similarity matrix. 
    If "user", it is Pearson similarity matrix; 
    if "item", it is adjusted cosine similarity matrix.
    :param k: number of neighbors who is considered the most similar to the target id.
    :return predicted ratings: a pandas.Series, predicted ratings for the target item or user
    """ 
    # drop the target itself
    similarities = similarity_matrix[id].drop(id)
    neighbors = similarities.nlargest(k).index
    neighbors_sim = similarities.loc[neighbors].values
    weights_sum = sum(neighbors_sim)

    # avoid division by zero
    if weights_sum == 0:
        return pd.Series(dtype='float64')  # return an empty Series: Series([], dtype: float64)
    
    if user_or_item == 'user':
        neighbors_ratings = utility_matrix.loc[neighbors]
        weighted_ratings = neighbors_ratings.mul(neighbors_sim, axis=0)
        predicted_ratings = weighted_ratings.sum(axis=0) / weights_sum
    elif user_or_item == 'item':
        neighbors_ratings = utility_matrix[neighbors]
        weighted_ratings = neighbors_ratings.mul(neighbors_sim, axis=1)
        predicted_ratings = weighted_ratings.sum(axis=1) / weights_sum
    return predicted_ratings
    
def recommend_top_n(user_id, utility_matrix, similarity_matrix, n, k):
    """
    Recommend top n items for the user.
    User-User CF.

    :param user_id: user id
    :param utility_matrix: utility matrix
    :param similarity_matrix: similarity matrix, Pearson similarity matrix here.
    :param n: number of items to recommend
    :param k: number of neighbors who is considered the most similar to the target user.
    :return: list of top n items to recommend
    """
    predicted_ratings = predict_for_target_id(user_id, 'user', utility_matrix, similarity_matrix, k)
    
    # deal with empty predicted ratings
    if predicted_ratings.empty:
        return []
    
    # remove items that the user has already seen
    user_seen = utility_matrix.loc[user_id].dropna().index
    predicted_ratings = predicted_ratings.drop(user_seen, errors='ignore')

    top_k_items = predicted_ratings.nlargest(n).index.tolist()
    return top_k_items

def item_based_recommend_top_n(user_id, utility_matrix, similarity_matrix, n, k):
    """
    Recommend top n items for the user.
    Item-Item CF.

    :param user_id: user id
    :param utility_matrix: utility matrix
    :param similarity_matrix: similarity matrix, adjusted cosine similarity matrix here.
    :param n: number of items to recommend
    :param k: number of neighbors who is considered the most similar to the target user.
    :return: list of top n items to recommend
    """
    # drop those movies rated by the user before (in the training set)
    user_rated = utility_matrix.loc[user_id].dropna().index
    candidate_items = utility_matrix.columns.difference(user_rated)

    predictions = {}

    for item_id in candidate_items:
        # Similar to part of 'predict_for_target_id' function
        similarities = similarity_matrix[item_id].drop(item_id, errors='ignore')

        # Tips: Only consider those items that the user has rated is enough.
        similarities = similarities.loc[similarities.index.intersection(user_rated)]

        if similarities.empty:
            continue

        neighbors = similarities.nlargest(k).index
        neighbors_sim = similarities.loc[neighbors].values
        weights_sum = sum(neighbors_sim)

        # avoid division by zero
        if weights_sum == 0:
            continue

        # neighbors_ratings_for_curr_user = utility_matrix.loc[user_id, neighbors].fillna(0)
        # since we use "intersection" above, no need for fillna(0) here
        neighbors_ratings_for_curr_user = utility_matrix.loc[user_id, neighbors]
        # weighted_ratings = neighbors_ratings_for_curr_user.mul(neighbors_sim, axis=1)
        weighted_ratings = neighbors_ratings_for_curr_user * neighbors_sim
        predicted_rating = weighted_ratings.sum() / weights_sum
        predictions[item_id] = predicted_rating

    if not predictions:
        return []
    
    pred_series = pd.Series(predictions)
    top_k_items = pred_series.nlargest(n).index.tolist()
    return top_k_items

def recall_n(test_df, recommendations):
    """
    Calculate recall@N.

    :param test_df: pandas.DataFrame, test data
    :param recommendations: dictionary, user_id -> recommended items (list of item_ids)
    :return: recall@N score
    """
    hit = 0
    total = 0
    # group by userId, get the set of movies watched by each user
    grouped = test_df.groupby('userId')['movieId'].apply(set)

    for user_id, watched_movies in grouped.items():
        recommended_movies = recommendations.get(user_id, [])
        hit += len(set(recommended_movies) & watched_movies)
        total += len(watched_movies)
    
    if total == 0:
        return 0
    else:
        return hit / total


def build_movie_profiles(movies_df):
    """
    Build movie profiles using TF-IDF.

    :param movies_df: pandas.DataFrame, movies data
    :return: movie profiles (TF-IDF matrix)
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))
    # print(tfidf.get_feature_names_out())
    movies_profiles = pd.DataFrame(tfidf_matrix.toarray(), index=movies_df['movieId'])

    return movies_profiles


def content_based_top_n(user_id, utility_matrix, movie_profiles, n):
    """
    Content-based recommendation for top n items.

    :param user_id: user id
    :param utility_matrix: utility matrix
    :param movie_profiles: movie profiles (TF-IDF matrix)
    :param n: number of items to recommend
    :return: list of top n items to recommend
    """
    user_ratings = utility_matrix.loc[user_id].dropna()
    if user_ratings.empty:
        return []
    
    # build user profile
    rated_profiles = movie_profiles.loc[user_ratings.index]
    user_profile = np.dot(user_ratings.values, rated_profiles.values) / user_ratings.sum()  # numPy 1-D array

    # calculate cosine similarity between user profile and movie profiles
    # [user_profile]: to make it 2-D array
    # cosine_similarity(A, B) output: A.shape[0] x B.shape[0]
    # [0]: to get the only one row of the result
    similarities = cosine_similarity([user_profile], movie_profiles.values)[0]
    sim_series = pd.Series(similarities, index=movie_profiles.index)
    # remove items that the user has already seen
    sim_series = sim_series.drop(user_ratings.index, errors='ignore')
    top_k_items = sim_series.nlargest(n).index.tolist()
    return top_k_items

def hybrid_top_n(user_id, utility_matrix, user_similarity_matrix, movie_profiles, n, k, alpha=0.5):
    """
    Hybrid recommendation for top n items.

    :param user_id: user id
    :param utility_matrix: utility matrix
    :param user_similarity_matrix: user similarity matrix (Pearson similarity matrix)
    :param movie_profiles: movie profiles (TF-IDF matrix)
    :param n: number of items to recommend
    :param k: number of neighbors who is considered the most similar to the target user.
    :param alpha: weight for CF and content-based recommendation
    :return: list of top n items to recommend
    """
    cf_scores = predict_for_target_id(user_id, 'user', utility_matrix, user_similarity_matrix, k)
    
    # deal with empty predicted ratings
    if cf_scores.empty:
        return []
    
    # remove items that the user has already seen
    user_seen = utility_matrix.loc[user_id].dropna().index
    cf_scores = cf_scores.drop(user_seen, errors='ignore')

    # build user profile using content-based method
    user_ratings = utility_matrix.loc[user_id].dropna()
    rated_profiles = movie_profiles.loc[user_ratings.index]
    user_profile = np.dot(user_ratings.values, rated_profiles.values) / user_ratings.sum()

    # calculate cosine similarity between user profile and movie profiles
    similarities = cosine_similarity([user_profile], movie_profiles.values)[0]
    sim_series = pd.Series(similarities, index=movie_profiles.index)
    
    # remove items that the user has already seen
    sim_series = sim_series.drop(user_seen, errors='ignore')
    
    # combine CF and content-based scores using alpha parameter
    combined_scores = alpha * cf_scores.add((1 - alpha) * sim_series, fill_value=0)
    
    top_k_items = combined_scores.nlargest(n).index.tolist()
    
    return top_k_items


def main():
    df = pd.read_csv('ml-small/ratings.csv')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    user_ids = test_df['userId'].unique()

    N = 200
    # N = 1000

    # Build the utility matrix
    utility_matrix = build_utility_matrix(train_df)
    # User-User CF
    user_similarity_matrix = pearson_similarity(utility_matrix)
    # print(user_similarity_matrix)
    user_user_CF_recommendations = {}
    for user_id in user_ids:
        user_user_CF_recommendations[user_id] = recommend_top_n(user_id, utility_matrix, user_similarity_matrix, n=N, k=200)

    user_user_recall = recall_n(test_df, user_user_CF_recommendations)
    print(f"User-User CF Recall: {user_user_recall:.4f}")

    # Item-Item CF
    # movie_similarity_matrix = adjusted_cosine_similarity(utility_matrix)
    # # print(movie_similarity_matrix)
    # item_item_CF_recommendations = {}
    # for user_id in tqdm(user_ids):
    #     item_item_CF_recommendations[user_id] = item_based_recommend_top_n(user_id, utility_matrix, movie_similarity_matrix, n=N, k=1000)

    # item_item_recall = recall_n(test_df, item_item_CF_recommendations)
    # print(f"Item-Item CF Recall: {item_item_recall:.4f}")

    # Content-based recommendation
    movies_df = pd.read_csv('ml-small/movies.csv')
    movie_profiles = build_movie_profiles(movies_df)
    content_based_recommendations = {}
    for user_id in user_ids:
        content_based_recommendations[user_id] = content_based_top_n(user_id, utility_matrix, movie_profiles, n=N)
    
    content_based_recall = recall_n(test_df, content_based_recommendations)
    print(f"Content-based Recall: {content_based_recall:.4f}")

    # Hybrid recommendation
    hybrid_recommendations = {}
    for user_id in user_ids:
        hybrid_recommendations[user_id] = hybrid_top_n(user_id, utility_matrix, user_similarity_matrix, movie_profiles, n=N, k=200, alpha=0.5)

    hybrid_recall = recall_n(test_df, hybrid_recommendations)
    print(f"Hybrid Recall: {hybrid_recall:.4f}")


if __name__ == '__main__':
    main()