import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

##Loading a data from CSV file
categories = ['Business', 'Health', 'Sports']
df = pd.read_csv('D:\Master Course\Task 2_Clustering/document.csv', usecols=['Text','Class'], encoding='latin1')
category_counts = df['Class'].value_counts()
print(category_counts)

##Filetring the data
data = []
for category in categories:
    category_data = df[df['Text'] == category.upper()]
    # Append the category data to the list
    data.append(category_data)

# Concatenate the category data into a single DataFrame
df = pd.concat(data)

filtered_category_counts = df['Text'].value_counts()
print(filtered_category_counts)

null_rows = df[df.isnull().any(axis=1)]

# Display null rows
print(f'Null rows:\n{null_rows}')

print(df.head())
print(df.tail())

# If your data is stored in a file (e.g., CSV), you'll need to load it into a DataFrame. You can do this using Pandas:
df = pd.read_csv('document.csv')

# Concatenate columns: Concatenate the 'Text' and 'Class' columns.
df['Text'] = df['Class'] + ' ' + df['Text']

# IInitialize TF-IDF vectorizer: Initialize the TfidfVectorizer with the desired parameters.
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)

# Fit the vectorizer: Fit the vectorizer on the concatenated text.
X = vectorizer.fit_transform(df['Text'])

# Print TF-IDF matrix shape: Print the shape of the TF-IDF matrix to verify the result
print("TF-IDF matrix shape:", X.shape)

##Elbow method vs Silhouette score method:

# Define range of clusters to evaluate
range_n_clusters = range(2, 6)
distortions = []
silhouette_scores = []

# Calculate distortions and silhouette scores for each number of clusters
for n_clusters in range_n_clusters:
    kmodel = KMeans(n_clusters=n_clusters, random_state=42)
    kmodel.fit(X)
    distortions.append(kmodel.inertia_)
    silhouette_scores.append(silhouette_score(X, kmodel.labels_))

# Plot results
    
labels = [f'Cluster {i}' for i in range_n_clusters]
sizes_distortion = distortions
sizes_silhouette = silhouette_scores

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.pie(sizes_distortion, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Distortion Proportion')

plt.subplot(1, 2, 2)
plt.pie(sizes_silhouette, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Silhouette Score Proportion')

plt.tight_layout()
plt.show()

# KMeans clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=10, random_state=42)
kmeans.fit(X)

# Assuming kmeans is your fitted KMeans model
unique_clusters = len(set(kmeans.labels_))

print("Number of clusters:", unique_clusters)

unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)

# Print unique labels and their counts
for label, count in zip(unique_labels, counts):
    print(f"Cluster: {label}, Count: {count} documents")


# Reduce dimensionality for visualization
    
# Get the number of points in each cluster
cluster_sizes = [np.sum(kmeans.labels_ == i) for i in range(n_clusters)]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
tfidf_reduced = pca.fit_transform(X.toarray())
# Plot clusters with sizes proportional to the number of points
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(tfidf_reduced[kmeans.labels_ == i, 0], tfidf_reduced[kmeans.labels_ == i, 1], 
                s=cluster_sizes[i]*10, alpha=0.5, label=f'Cluster {i + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=100, c='black', label='Centroids')
plt.title('KMeans Clustering of Text Data Class')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()


##find unique category with cluster number
labels = kmeans.labels_
wiki_cl=pd.DataFrame(list(zip(df['Class'],labels)),columns=['Class','cluster']) 
unique_category_cluster = wiki_cl.drop_duplicates()
print(unique_category_cluster)

##predict Cluster Method

def get_category(cluster):
    return {
        0: 'Business',
        1: 'Health',
        2: 'Sports'
    }.get(cluster, 'Unknown')

def predict_cluster_and_category(input_text):
    input_vector = vectorizer.transform([input_text])
    print(input_vector)
    cluster = kmeans.predict(input_vector)[0]
    category = df.iloc[cluster]['Class']
    
    print("Predicted cluster:", cluster)

    return cluster, get_category(cluster)

#Graphical User Interface

# Tkinter GUI
window = tk.Tk()
window.title("Enter the text to be categorized")
window.minsize(600, 400)

text_box = ScrolledText(window)
text_box.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

predicted_cluster_label = tk.Label(window, text="Predicted Score:")
predicted_cluster_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

predicted_category_label = tk.Label(window, text="Predicted Class:")
predicted_category_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

def process_input():
    input_text = text_box.get("1.0", tk.END)
    predicted_cluster, category = predict_cluster_and_category(input_text)
    predicted_cluster_label.config(text=f"Predicted Score: {predicted_cluster}")
    predicted_category_label.config(text=f"Predicted Class: {category}")
    print("Predicted Title:", predicted_cluster)
    print("Predicted Class:", category)

# Tkinter GUI components
btn = tk.Button(window, text="Check Class", command=process_input)
btn.grid(row=3, column=0, padx=5, pady=5)

window.mainloop()








