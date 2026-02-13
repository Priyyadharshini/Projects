import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

data = pd.merge(ratings, movies, on="movieId")

movie_user_matrix = data.pivot_table(index="title",
                                     columns="userId",
                                     values="rating").fillna(0)

similarity = cosine_similarity(movie_user_matrix)

pickle.dump(movie_user_matrix, open("matrix.pkl","wb"))
pickle.dump(similarity, open("similarity.pkl","wb"))

print("Model saved")
