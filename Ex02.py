from sklearn import cluster, datasets, metrics
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


X, y = make_moons(n_samples=1000, noise=0.2, random_state=10)

# sp = plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
# plt.xlabel('X', fontsize=20)
# plt.ylabel('y', fontsize=20)
# plt.show()

# standardizing dataset to make it behave more like a normal distributed set
# ML estimators behave more nicely when sets are normal distributed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, n_init='auto', random_state=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# calculating silhouette score given various number of clusters
# higher score means better fitting
num_clusters = [2, 3, 4]
sil = []
max_sil = 0.0
for i in num_clusters:
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    sil_score = metrics.silhouette_score(X_scaled, labels)
    sil.append(sil_score)
    print('score for ', i, ' clusters is ', sil_score)
    if sil_score > max_sil:
        max_sil = sil_score
        best_k = i

print('max score found at ', best_k, ' clusters')
plt.plot(num_clusters, sil)
plt.xlabel('num clusters')
plt.ylabel('sil score')
plt.show()
