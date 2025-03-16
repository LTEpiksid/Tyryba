# k_nn_klasifikacija.py
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def run_knn(feature1="runtime_minutes", feature2="movie_averageRating", num_neighbors=3, rating_threshold=7):
    """
    Įkelia „movie_statistic_dataset.csv“ failą iš CSV aplanko, sukuria dvejetainį tikslą “rating_class”
    pagal 'movie_averageRating' (jei reitingas mažesnis už slenkstį – 0, kitaip – 1), pasirenka du požymius
    (pagal kur bus pavaizduota 2D erdvėje), vykdo k-NN klasifikaciją, braižo sprendimų ribas ir grąžina
    testavimo tikslumą bei sukurta Matplotlib figūrą.

    Parametrai:
      feature1, feature2 – dviejų požymių pavadinimai;
      num_neighbors – k reikšmė;
      rating_threshold – reitingo slenkstis.

    Grąžina žodyną su "num_neighbors", "test_accuracy" ir "figure".
    """
    csv_path = os.path.join(os.path.dirname(__file__), "CSV", "movie_statistic_dataset.csv")
    df = pd.read_csv(csv_path)

    # Sukuriame dvejetainį tikslą: rating_class, 0 jei movie_averageRating < slenkstis, kitaip 1.
    df["rating_class"] = df["movie_averageRating"].apply(lambda x: 0 if x < rating_threshold else 1)

    X = df[[feature1, feature2]].to_numpy()
    y = df["rating_class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean')
    knn.fit(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    # Sukuriame tinklelio duomenis braižyti sprendimų ribas.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', label='Treniravimo')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='^', edgecolor='k', label='Testavimo')
    ax.set_title(f"k-NN klasifikacija (k = {num_neighbors})\nTestavimo tikslumas: {test_accuracy:.2f}")
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.legend()

    result = {"num_neighbors": num_neighbors, "test_accuracy": test_accuracy, "figure": fig}
    return result


if __name__ == "__main__":
    res = run_knn()
    res["figure"].show()
