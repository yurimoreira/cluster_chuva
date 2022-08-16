from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from dtaidistance import dtw

filename = 'Paranatinga II'
est = read_csv(filename+'.csv', sep=',', parse_dates=True, index_col='data')
est.dropna(inplace=True, axis=1, how='all')
est = est[~est[est.columns[-1]].isna()].fillna(0)

#juntando a chuva mensal
est_month = est.resample('M').sum()

#matriz ditancia
dist_matrix = DataFrame(dtw.distance_matrix_fast(est_month.T.values), index=est_month.columns, columns=est_month.columns)
#figurinha legal pra ver os clusters
plt.figure(figsize=(16,9))
plt.title(f"Dendrogram for ward-linkage with correlation distance")
dn = dendrogram(linkage(est_month.T, method='ward'), distance_sort=True, above_threshold_color='blue')
#montando os cluesters
cluster_labels = fcluster(linkage(est_month.T, method='ward'), 20, criterion='maxclust')
cluster = dict(zip(est_month.columns, cluster_labels))

output = est_month.T.rename(cluster).groupby(axis=0, level=0).mean().T

filename_out = filename + '_cluster.csv'
output.to_csv(filename_out, sep=',')
