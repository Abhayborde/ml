import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
X = df[['Experience', 'Age']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for i in range(1, 6):  # Ensure number of clusters is <= number of samples
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 6), inertia, marker='o')  # Ensure number of clusters is <= number of samples
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust number of clusters to match dataset
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_means = df.groupby('Cluster')[['Experience', 'Age']].mean()
print(cluster_means)
