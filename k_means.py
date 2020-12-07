from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow(X_train):
    wcss = []
    for i in range(1,20):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1000, random_state=0, n_init=10)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_) # Sum of squared distances of samples to their closest cluster center.

    plt.plot(range(1,20),wcss)
    plt.title("The Elbow Method")
    plt.ylabel("no. of clusters")
    plt.xlabel("WCSS")
    plt.show()

def kmeans_fit_predict(X_train, X_test):
    kmeans = KMeans(n_clusters=10, init="k-means++", max_iter=1000, random_state=0, n_init=10)
    kmeans.fit(X_train)
    return kmeans.predict(X_test)