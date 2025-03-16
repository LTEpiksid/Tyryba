# k_means_clasterizacija.py
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")  # Naudojame TkAgg backend, kad dirbtų su Tkinter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def run_k_means(selected_features=None, num_clusters=3):
    """
    Įkelia „movie_statistic_dataset.csv“ failą iš CSV aplanko, atlieka 'genres' stulpelio one‑hot
    kodavimą, pasirenka požymius (jeigu nenustatyta – pagal numatytuosius skaitmeninius požymius ir
    one‑hot užkoduotus genres), vykdo k‑Means klasterizaciją, mažina duomenų matmenis naudojant PCA
    bei sukuria Matplotlib figūrą su klasterių rezultatais.

    Parametrai:
      selected_features – požymių sąrašas (jei None, naudojami numatytieji);
      num_clusters – klasterių skaičius.

    Grąžina žodyną, kuriame yra:
      "num_clusters", "silhouette_score", "inertia" ir "figure" – sukurta Matplotlib figūra.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "CSV", "movie_statistic_dataset.csv")
    df = pd.read_csv(csv_path)

    # One-hot kodavimas 'genres' stulpeliui.
    if "genres" in df.columns:
        genres_dummies = df["genres"].str.get_dummies(sep=",")
        if "\\N" in genres_dummies.columns:
            genres_dummies = genres_dummies.drop(columns=["\\N"])
        df = pd.concat([df, genres_dummies], axis=1)
        df = df.drop(columns=["genres"])
    else:
        raise KeyError("Duomenų faile trūksta stulpelio 'genres'!")

    # Jeigu nepateikta, numatytieji požymiai: skaitmeniniai stulpeliai + one-hot genres
    if selected_features is None:
        numeric_features = [
            "runtime_minutes", "movie_averageRating", "movie_numerOfVotes",
            "approval_Index", "Production budget $", "Domestic gross $", "Worldwide gross $"
        ]
        genre_features = list(genres_dummies.columns)
        selected_features = numeric_features + genre_features

    X = df[selected_features].to_numpy()

    # Vykdoma k‑Means klasterizacija.
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    sil_score = silhouette_score(X, labels) if num_clusters > 1 else None

    # PCA: mažiname duomenų matmenis iki 2, kad galėtume pavaizduoti diagramą.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], color='red', s=200, marker='X')
    title = f"K-Means klasterizacija (k = {num_clusters})"
    if sil_score is not None:
        title += f"\nSilueto rodiklis: {sil_score:.2f}"
    ax.set_title(title)
    ax.set_xlabel("Pagrindinė komponentė 1")
    ax.set_ylabel("Pagrindinė komponentė 2")

    result = {"num_clusters": num_clusters, "silhouette_score": sil_score, "inertia": kmeans.inertia_, "figure": fig}
    return result


if __name__ == "__main__":
    res = run_k_means()
    res["figure"].show()
