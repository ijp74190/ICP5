#Ian Pope 700717419
#ICP 5

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#pd.set_option('display.max_columns', 10)
#pd.set_option('display.width', 250)


#QUESTION 1
#Read in CC General data set and fill NaN values with 0
ccgen = pd.read_csv('CC GENERAL.csv')
#print(ccgen)
dataset = ccgen.fillna(0)

#Gets the dataset without the customer id
x = dataset.drop("CUST_ID",axis=1)
#print(x)

'''
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
'''
nclusters = 3

#Raw data
#Fit the k-means
km = KMeans(n_clusters=nclusters)
km.fit(x)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("The raw data's silhouette score is:", score)

#PCA data
pca = PCA(3)
x_pca = pca.fit_transform(x)
#Fit and run the k-means
km.fit(x_pca)
y_cluster_kmeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print("The PCA data's silhouette score is:", score)

#Scale x
scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)

#Scaled Data
km.fit(x_scaler)
y_cluster_kmeans = km.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
print(" Scaled data's silhouette score is:", score)


#PCA Scaled data
pca = PCA(3)
x_pca = pca.fit_transform(x_scaler)
#Fit and run the k-means
km.fit(x_pca)
y_cluster_kmeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print("ScaledPCA data silhouette score is:", score)
print()


#QUESTION 2
speech_data = pd.read_csv('pd_speech_features.csv')
x = speech_data.drop("class",axis=1)
y = speech_data["class"]

#Apply scalar
scaler = StandardScaler()
scaler.fit(x)
speech_scalar = scaler.transform(x)

#Apply PCA
pca = PCA(3)
x_pca = pca.fit_transform(speech_scalar)
df2 = pd.DataFrame(data=x_pca)

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 0)

#Run SVM
svc = LinearSVC(dual=False)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

#Display results
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("svm accuracy of training data=", acc_svc)
from sklearn.metrics import accuracy_score
acc_svc = round(accuracy_score(y_pred,y_test) * 100, 2)
print('accuracy of test data=',acc_svc)
print()


#QUESTION 3
iris = pd.read_csv('Iris.csv')
x = iris.iloc[:,[1,2,3,4]]
y = iris.iloc[:,-1]

#Apply LDA of compnents = 2
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x,y)
tlda = pd.DataFrame(lda.transform(x))
print(tlda)

#QUESTION 4
print("\nBoth PCA and LDA are methods that aim to perform dimensionality reduction")
print("PCA is an unsupervised technique that transforms the data into a number of")
print("\tprincipal components in the form of eigenvectors of a maximum variance")
print("LDA is a supervised technique that aims to maximize class seperability")
print("\tLDA uses linear discriminants to find which features best seperate the data")
