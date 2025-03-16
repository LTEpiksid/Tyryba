# pagrindinis_langas.py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import k_means_clasterizacija
import k_nn_klasifikacija


def run_methods():
    # Išvalome ankstesnius rezultatus.
    for widget in results_frame.winfo_children():
        widget.destroy()
    txt_compare.delete(1.0, tk.END)
    lbl_info.config(text="")

    method = method_var.get()

    # Gauti parametrus iš GUI.
    try:
        km_clusters = int(km_clusters_entry.get())
    except:
        km_clusters = 3
    km_features_input = km_features_entry.get().strip()
    if km_features_input:
        km_features = [f.strip() for f in km_features_input.split(",")]
    else:
        km_features = None

    try:
        knn_neighbors = int(knn_neighbors_entry.get())
    except:
        knn_neighbors = 3
    knn_feature1 = knn_feature1_var.get()
    knn_feature2 = knn_feature2_var.get()
    try:
        rating_threshold = float(rating_threshold_entry.get())
    except:
        rating_threshold = 7.0

    if method == "k-means":
        km_res = k_means_clasterizacija.run_k_means(selected_features=km_features, num_clusters=km_clusters)
        canvas = FigureCanvasTkAgg(km_res["figure"], master=results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        info = f"K-Means klasterizacija\nKlasterių skaičius: {km_res['num_clusters']}\n"
        if km_res["silhouette_score"] is not None:
            info += f"Silueto rodiklis: {km_res['silhouette_score']:.2f}\n"
        info += f"Inercija: {km_res['inertia']:.2f}"
        lbl_info.config(text=info)
    elif method == "k-NN":
        knn_res = k_nn_klasifikacija.run_knn(feature1=knn_feature1, feature2=knn_feature2,
                                             num_neighbors=knn_neighbors, rating_threshold=rating_threshold)
        canvas = FigureCanvasTkAgg(knn_res["figure"], master=results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        info = f"k-NN klasifikacija\nk reikšmė: {knn_res['num_neighbors']}\nTestavimo tikslumas: {knn_res['test_accuracy']:.2f}"
        lbl_info.config(text=info)
    elif method == "both":
        km_res = k_means_clasterizacija.run_k_means(selected_features=km_features, num_clusters=km_clusters)
        knn_res = k_nn_klasifikacija.run_knn(feature1=knn_feature1, feature2=knn_feature2,
                                             num_neighbors=knn_neighbors, rating_threshold=rating_threshold)
        # Sukuriame dvi sub-langas šalia vienas kito.
        left_frame = tk.Frame(results_frame)
        right_frame = tk.Frame(results_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        right_frame.pack(side="right", fill="both", expand=True)

        canvas1 = FigureCanvasTkAgg(km_res["figure"], master=left_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        canvas2 = FigureCanvasTkAgg(knn_res["figure"], master=right_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Paruošiame palyginimo paaiškinimą.
        compare_text = "Palyginimo rezultatai:\n\n"
        compare_text += f"k-NN tikslumas: {knn_res['test_accuracy']:.2f}\n"
        if km_res["silhouette_score"] is not None:
            compare_text += f"K-Means silueto rodiklis: {km_res['silhouette_score']:.2f}\n"
        else:
            compare_text += "K-Means silueto rodiklis: N/A\n"
        compare_text += f"K-Means inercija: {km_res['inertia']:.2f}\n\n"
        compare_text += (
            "Interpretacija:\n"
            " - k-NN yra prižiūrimas metodas: aukštas testavimo tikslumas rodo, kad modelis gerai atpažįsta\n"
            "   duomenų elementus pagal pasirinktą tikslą (žemesnis reitingas – 0, aukštesnis – 1).\n"
            " - K-Means yra neprižiūrimas klasterizacijos metodas: aukštesnis silueto rodiklis ir mažesnė inercija\n"
            "   rodo gerai atskirtas grupes.\n\n"
            "Šie metodai tarnauja skirtingiems tikslams. Jei reikia prognozuoti, k-NN gali būti tinkamesnis,\n"
            "o jei norima ištirti duomenų struktūrą – K-Means suteiks įžvalgų."
        )
        txt_compare.delete(1.0, tk.END)
        txt_compare.insert(tk.END, compare_text)
        lbl_info.config(text="Abu metodai parodyti šalia vienas kito:")


# Sukuriame pagrindinį langą.
root = tk.Tk()
root.title("Algoritmų testavimas")

# Metodų pasirinkimo dalis.
method_var = tk.StringVar(value="k-means")
lbl = tk.Label(root, text="Pasirinkite vykdytiną algoritmą:", font=("Arial", 12))
lbl.grid(row=0, column=0, sticky="w", padx=5, pady=5)
frame_methods = tk.Frame(root)
frame_methods.grid(row=1, column=0, sticky="w", padx=5)
tk.Radiobutton(frame_methods, text="K-Means klasterizacija", variable=method_var, value="k-means").pack(anchor="w")
tk.Radiobutton(frame_methods, text="k-NN klasifikacija", variable=method_var, value="k-NN").pack(anchor="w")
tk.Radiobutton(frame_methods, text="Abu (Palyginti)", variable=method_var, value="both").pack(anchor="w")

# K-Means parametrų nustatymo dalis.
km_frame = tk.LabelFrame(root, text="K-Means parametrų nustatymai", padx=5, pady=5)
km_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
tk.Label(km_frame, text="Klasterių skaičius:").grid(row=0, column=0, sticky="w")
km_clusters_entry = tk.Entry(km_frame, width=10)
km_clusters_entry.insert(0, "3")
km_clusters_entry.grid(row=0, column=1, padx=5, pady=2)
tk.Label(km_frame, text="Požymiai (atskiriami kableliu) [pasirinktinai]:").grid(row=1, column=0, sticky="w")
km_features_entry = tk.Entry(km_frame, width=40)
km_features_entry.insert(0, "")
km_features_entry.grid(row=1, column=1, padx=5, pady=2)

# k-NN parametrų nustatymo dalis.
knn_frame = tk.LabelFrame(root, text="k-NN parametrų nustatymai", padx=5, pady=5)
knn_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
numeric_options = ["runtime_minutes", "movie_averageRating", "movie_numerOfVotes",
                   "approval_Index", "Production budget $", "Domestic gross $", "Worldwide gross $"]
tk.Label(knn_frame, text="Požymis 1:").grid(row=0, column=0, sticky="w")
knn_feature1_var = tk.StringVar(value="runtime_minutes")
ttk.OptionMenu(knn_frame, knn_feature1_var, numeric_options[0], *numeric_options).grid(row=0, column=1, padx=5, pady=2)
tk.Label(knn_frame, text="Požymis 2:").grid(row=1, column=0, sticky="w")
knn_feature2_var = tk.StringVar(value="movie_averageRating")
ttk.OptionMenu(knn_frame, knn_feature2_var, numeric_options[1], *numeric_options).grid(row=1, column=1, padx=5, pady=2)
tk.Label(knn_frame, text="k reikšmė:").grid(row=2, column=0, sticky="w")
knn_neighbors_entry = tk.Entry(knn_frame, width=10)
knn_neighbors_entry.insert(0, "3")
knn_neighbors_entry.grid(row=2, column=1, padx=5, pady=2)
tk.Label(knn_frame, text="Reitingo slenkstis:").grid(row=3, column=0, sticky="w")
rating_threshold_entry = tk.Entry(knn_frame, width=10)
rating_threshold_entry.insert(0, "7")
rating_threshold_entry.grid(row=3, column=1, padx=5, pady=2)

# Rezultatų rodymo sritis.
results_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=2)
results_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)

# Informacinė etiketė.
lbl_info = tk.Label(root, text="", font=("Arial", 10))
lbl_info.grid(row=5, column=0, sticky="w", padx=5, pady=5)

# ScrolledText laukelis paaiškinimui (kai pasirenkami abu metodai).
txt_compare = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
txt_compare.grid(row=6, column=0, padx=5, pady=5)

# Mygtukas "Paleisti".
run_button = ttk.Button(root, text="Paleisti pasirinktą metodą (-as)", command=run_methods)
run_button.grid(row=7, column=0, pady=10)

root.mainloop()
