from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
sns.set(style="ticks")


def visualize_distribution_static(X):
    X=X.to_numpy()
    X_embedding = TSNE(n_components=2).fit_transform(X)
    # fig, ax = plt.subplots(1, 2, figsize=(15, 15), dpi=300,sharey=True)
    # plt.figure(1)
    # sns.jointplot(X_embedding[:,0], X_embedding[:,1], kind="hex", color="#4CB391")
    # plt.figure(2)
    sns_plot=sns.jointplot(X_embedding[:,0], X_embedding[:,1], kind="kde", space=0, color="#4CB391")
    sns_plot.savefig('./output/img/distribution.png')
    plt.show()



def visualize_distribution_time_serie(ts,value):
    ts = pd.DatetimeIndex(ts)
    value=value.to_numpy()[:,1:]
    data = pd.DataFrame(value,ts)
    # fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
    data = data.rolling(2).mean()
    sns_plot=sns.lineplot(data=data, palette="BuGn_r", linewidth=0.5)
    sns_plot.figure.savefig('./output/img/timeserie.png')
    # ax.set_ylabel('')
    # ax.set_xlabel('')
    plt.show()


def visualize_outlierscore(value,label,contamination):
    ts = np.arange(len(value))
    outlier_label=[]
    for i in range(len(ts)):
        if label[i]==1:
            outlier_label.append('inliner')
        else:
            outlier_label.append('outlier')
    X_outlier = pd.DataFrame({'ts':ts,'Outlier_score':value,'outlier_label':np.array(outlier_label)})
    pal = dict(inliner="#4CB391", outlier="gray")
    g = sns.FacetGrid(X_outlier, hue="outlier_label", palette=pal, height=5)
    g.map(plt.scatter, "ts", "Outlier_score", s=30, alpha=.7, linewidth=.5, edgecolor="white")

    ranking = np.sort(value)
    threshold = ranking[int((1 - contamination) * len(ranking))]
    plt.hlines(threshold, xmin=0, xmax=len(X_outlier)-1, colors="g", zorder=100, label='Threshold')
    plt.savefig('./output/img/visualize_outlierscore_time.png')
    plt.show()

def visualize_outlierresult(X,label):
    X['outlier']=pd.Series(label)
    pal = dict(inliner="#4CB391", outlier="gray")
    g = sns.pairplot(X, hue="outlier", palette=pal)
    plt.show()


