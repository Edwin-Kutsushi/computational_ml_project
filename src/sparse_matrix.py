from scipy.sparse import coo_matrix


def build_sparse_matrix(df):
    user_cat = df["userId"].astype("category")
    movie_cat = df["movieId"].astype("category")

    user_codes = user_cat.cat.codes
    movie_codes = movie_cat.cat.codes
    ratings = df["rating"].astype(float)

    sparse_matrix = coo_matrix(
        (ratings, (user_codes, movie_codes))
    ).tocsr()

    user_map = dict(enumerate(user_cat.cat.categories))
    movie_map = dict(enumerate(movie_cat.cat.categories))

    return sparse_matrix, user_map, movie_map