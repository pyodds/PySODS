from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
sns.set(style="ticks")


def visualize_distribution(X,prediction,score):
    sns.set(style="ticks")
    X=X.to_numpy()
    X_embedding = TSNE(n_components=2).fit_transform(X)
    sns_plot=sns.jointplot(X_embedding[:,1],X_embedding[:,0], kind="kde", space=0, color="#4CB391")
    sns_plot.savefig('./output/img/distribution.png')


def visualize_distribution_static(X,prediction,score):
    sns.set(style="darkgrid")

    X=X.to_numpy()
    X_embedding = TSNE(n_components=2).fit_transform(X)

    outlier_label=[]
    for i in range(len(X_embedding)):
        if prediction[i]==1:
            outlier_label.append('inlier')
        else:
            outlier_label.append('outlier')
    X_outlier = pd.DataFrame({'x_emb':X_embedding[:,0],'y_emb':X_embedding[:,1],'outlier_label':np.array(outlier_label),'score':np.array(score)})
    new_sns = sns.scatterplot(x="x_emb", y="y_emb",hue = "score", sizes =20, palette = 'BuGn_r',legend = False, data = X_outlier)
    new_sns.get_figure().savefig('./output/img/distribution_withoutlier.png')



def visualize_distribution_time_serie(ts,value):
    sns.set(style="ticks")

    ts = pd.DatetimeIndex(ts)
    value=value.to_numpy()[:,1:]
    data = pd.DataFrame(value,ts)
    data = data.rolling(2).mean()
    sns_plot=sns.lineplot(data=data, palette="BuGn_r", linewidth=0.5)
    sns_plot.figure.savefig('./output/img/timeserie.png')
    plt.show()



def visualize_outlierscore(value,label,contamination):
    sns.set(style="darkgrid")

    ts = np.arange(len(value))
    outlier_label=[]
    for i in range(len(ts)):
        if label[i]==1:
            outlier_label.append('inlier')
        else:
            outlier_label.append('outlier')
    X_outlier = pd.DataFrame({'ts':ts,'Outlier_score':value,'outlier_label':np.array(outlier_label)})
    pal = dict(inlier="#4CB391", outlier="gray")
    g = sns.FacetGrid(X_outlier, hue="outlier_label", palette=pal, height=5)
    g.map(plt.scatter, "ts", "Outlier_score", s=30, alpha=.7, linewidth=.5, edgecolor="white")

    ranking = np.sort(value)
    threshold = ranking[int((1 - contamination) * len(ranking))]
    plt.hlines(threshold, xmin=0, xmax=len(X_outlier)-1, colors="g", zorder=100, label='Threshold')
    threshold = ranking[int((contamination) * len(ranking))]
    plt.hlines(threshold, xmin=0, xmax=len(X_outlier)-1, colors="g", zorder=100, label='Threshold2')
    plt.savefig('./output/img/visualize_outlierscore.png')
    plt.show()



def visualize_outlierresult(X,label):
    X['outlier']=pd.Series(label)
    pal = dict(inlier="#4CB391", outlier="gray")
    g = sns.pairplot(X, hue="outlier", palette=pal)
    plt.show()


