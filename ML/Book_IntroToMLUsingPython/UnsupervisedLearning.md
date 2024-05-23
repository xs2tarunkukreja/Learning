# Unsupervised Learning
There is no output known.
In unsupervised learning, the learning algorithm is just shown the input data and asked to extract knowledge from this data.

# Types of Unsupervised Learning
Two Types: Transformation and Clustering

## Transformation
It create new representation of dataset that is easy for human or machine to understand.
Common Application: Dimensionality Reduction.
Another Application: Finding parts or component of data. E.g. Topic for document.

## Clustering
Partition data into distinct groups of similar items.

# Challenge in Unsupervised Learning
Challenge is evaluating whether the algorithm learned something useful.
We don't know what is the right output. 
There is no way to tell algorithm what we want.

So, it is used mainly in exploratory setting when data scientist want to understand data.
It is also used as Preprocessing step for supervised learning.

# Preprocessing and Scaling
NN and SVM are very sensitive to scale of data.
Different kind of Preprocessing - Data have 2 features - feature 1 X - (10-15) and feature 2 Y (1 to 9).
    Standard Scalar - for feature, mean should be 0 and variance should be 1. It doesn't ensure min and max value.
    Robust Scalar - focus on median and quartile. So, it ignore datapoints that are very different from others.
    MinMax Scalar - Between (0 and 1).
    Normalizer - feature vector has enclidean length = 1. Project data point on a circle or sphere with radius 1. It is used when only direction matters, not the length of feature vector.

## Applying Data Tranformation
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) // It find min and max of each feature.
X_train_scaled = scaler.transform(X_train) // all features are between 0 and 1.

X_test_scaled = scaler.transform(X_test) // Should transform test data as well. It may be out 0 and 1.

## Effect of Preprocessing on Supervised Learning
It improve the performance/accuracy %.

# Dimensionality Reduction, Feature Extraction and Manifold Learning
One of the simplest and used algorithm is "Principal Component Analysis"
Others - "Non-Negative Matrix factorization" - Used for feature extraction
         "t-SNE" - Used for visualization.

## Principle Component Analysis (PCA)
Rotate the dataset in a way rotated feature are statistically uncorrelated. After rotation, selecting only subset of new features, according to how important they are for explaining the data.
Original Data > Transformed Data (first compoent align x and 2nd to y)> Transformed Data without first component > Back rotation using only first component. 

Breast Cancer Dataset have lots of features. So, we draw histogram for the features for 2 classes benign and malign.
    If overlap, means not informative.
This plots doesn't show interaction between variables.

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled) // find 2 principal components
X_pca = pca.transform(X_scaled) // apply rotation and dimensionality reduction. 
// By default, PCA only rotates but keep all principal components.

X_pca - It have only 2 featurs.

### Eigenfaces for feature extraction

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

// Data have images of famous person. Each person have several images.

// Take only 50 photos max per person
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

// scale the grayscale values to be between 0 and 1
// instead of 0 and 255 for better numeric stability
X_people = X_people / 255.

Now we need to find the people name based on image.
Classifier - each person as a class.
A simple solution is to use a one-nearest-neighour classifier

from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
// Test set score of 1-nn: 0.27

Computing distances in the original pixel space is quite a bad way to measure similarity between faces. face can lie on any side of image.

Using distance along principal components can improve the accuracy.

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test))) = 36%.

we can guess which aspects of the face images some of the components are capturing. The first component seems to mostly encode the contrast between the face and the background, the second component encodes differences in  lighting between the right and the left half of the face, and so on.

## Non-Negative Matrix Factorization (NMF)
In PCA, we try to write each data point as a weighted sum of some components.

But whereas in PCA we wanted components that were orthogonal and that explained as much variance of the data as possible, in NMF, we want the components and the coefficients to be nonnegative; that is, we want both the components and the coefficients to be greater than or equal to zero.

So, NMF can be applied with data where each feature >= 0.
    Many people speaking
    Music from different instrument.
### Applying NMF on face images
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

## Manifold Learning with t-SNE
Manifold learning algorithms are mainly aimed at visualization, and so are rarely used to generate more than two new features.
<<<Need to repeat this...>>>

# Clustering
A task of partitioning the dataset into groups, called cluster.
## K-Mean Clustering
Find clusters center.
Assign each data points to cluster based on distance between points and center. (choose near one.)
Now find new center based on mean of all data point in cluster.

Repeat until assignment to cluster doen't change.

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
// generate synthetic two-dimensional data
X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

Clustering is somewhat similar to classification, in that each item gets a label. However, there is no ground truth, and consequently the labels themselves have no a priori meaning.

Running the algorithm again might result in a different numbering of clusters because of the random nature of the initialization.

### Failure cases for k-means
Each cluster is defined solely by its center, which means that each cluster is a convex shape. As a result of this, k-means can only capture relatively simple shapes.

### +ve, -ve
it is relatively easy to understand and implement, 
it runs relatively quickly. kmeans scales easily to large datasets.

One of the drawbacks of k-means is that it relies on a random initialization, which means the outcome of the algorithm depends on a random seed.
relatively restrictive assumptions made on the shape of clusters.
requirement to specify the number of clusters.

## Agglomerative Clustering
the algorithm starts by declaring each point its own cluster, and then merges the two most similar clusters until some stopping criterion is satisfied.
Stopping criteria may be number of clusters.

There are several linkage criteria that specify how exactly the “most similar cluster” is
measured.
    ward - The default choice, ward picks the two clusters to merge such that the variance within all 
        clusters increases the least. This often leads to clusters that are relatively equally sized.
    average - average linkage merges the two clusters that have the smallest average distance
        between all their points.
    complete - complete linkage (also known as maximum linkage) merges the two clusters that
        have the smallest maximum distance between their points.

Note - ward works on most datasets.

agglomerative clustering cannot make predictions for new data points. Therefore, Agglomerative Clustering has no predict method.

from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
### Hierarchical Clustering And Dendrograms
Agglomerative clustering produces what is known as a hierarchical clustering. The clustering proceeds iteratively, and every point makes a journey from being a single point cluster to belonging to some final cluster.

Dendrogram is a tool to visualize the same.


### +ve, -ve
agglomerative clustering still fails at separating complex shapes like the two_moons dataset.

## DBSCAN
densitybased spatial clustering of applications with noise.
The main benefits - it does not require the user to set the number of clusters a priori, it can capture
clusters of complex shapes, and it can identify points that are not part of any cluster. 
DBSCAN is somewhat slower than agglomerative clustering and k-means, but still scales to relatively large datasets.

DBSCAN works by identifying points that are in “crowded” regions of the feature space, where many data points are close together. These regions are referred to as dense regions in feature space. 

## Compare and Evaluate Clustering Algorithm
