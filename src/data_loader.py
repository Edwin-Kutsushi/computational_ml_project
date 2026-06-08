import pandas as pd
from pathlib import Path


def load_movielens_data(data_dir):
    data_dir = Path(data_dir)

    user_rating_history = pd.read_csv(data_dir / "user_rating_history.csv")
    ratings_additional = pd.read_csv(data_dir / "ratings_for_additional_users.csv")
    movies = pd.read_csv(data_dir / "movies.csv")

    return user_rating_history, ratings_additional, movies


def preprocess_movielens(user_rating_history, ratings_additional, movies):
    user_rating_history = user_rating_history[
        ["userId", "movieId", "rating", "tstamp"]
    ].copy()

    ratings_additional = ratings_additional[
        ["userId", "movieId", "rating", "tstamp"]
    ].copy()

    movies = movies[
        ["movieId", "title", "genres"]
    ].copy()

    ratings_all = pd.concat(
        [user_rating_history, ratings_additional],
        ignore_index=True
    )

    ratings_all = ratings_all.dropna(subset=["userId", "movieId", "rating"])
    ratings_all = ratings_all[ratings_all["rating"] >= 0].copy()

    valid_ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ratings_all = ratings_all[
        ratings_all["rating"].isin(valid_ratings)
    ].copy()

    ratings_all["tstamp"] = pd.to_datetime(
        ratings_all["tstamp"], errors="coerce"
    )

    ratings_all = ratings_all.dropna(subset=["tstamp"])

    ratings_all["userId"] = ratings_all["userId"].astype(int)
    ratings_all["movieId"] = ratings_all["movieId"].astype(int)

    ratings_all = ratings_all.sort_values("tstamp")

    ratings_all = ratings_all.drop_duplicates(
        subset=["userId", "movieId"],
        keep="last"
    ).copy()

    final_df = ratings_all.merge(
        movies,
        on="movieId",
        how="left"
    )

    final_dataset = final_df[
        ["userId", "movieId", "rating", "tstamp", "title", "genres"]
    ].copy()

    return final_dataset


def save_dataset(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)




from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=128):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='../data/mnist',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='../data/mnist',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader   