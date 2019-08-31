import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.cluster import AgglomerativeClustering

dframe1 = pd.read_table('Baby_Baby_Diapers_40Groups_41Topics.csv', sep=',')
dframe2 = pd.read_table('Baby_combined.csv', sep=',')

dframe = pd.merge(dframe1[["itemid"]], dframe2, on='itemid', how='left')
dframe = dframe[["itemid","views","clicks","rating_good","rating_normal","rating_bad"]]

dframe = dframe.replace([np.inf, -np.inf], np.nan)
dframe = dframe.dropna()

dframe_scaled = StandardScaler().fit_transform(dframe.iloc[:,1:6])

# Compute K-means
range_n_clusters = [3]
for n_c in range_n_clusters:
    # Number of clusters
    km = KMeans(n_clusters=n_c)
    # Fitting the input data
    km_res = km.fit(dframe_scaled)
    # Getting the cluster labels
    labels = km_res.predict(dframe_scaled)
    
    silhouette_avg = silhouette_score(dframe_scaled, labels)
    with open("result.txt", 'a') as myfile:
        myfile.write("For n_clusters = "+str(n_c)+", The silhouette_score is : "+str(silhouette_avg)+'\n')
    
    dframe["class"] = labels

dframe.to_csv('class_result.csv', sep=',', index=False)
