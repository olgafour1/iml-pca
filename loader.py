from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
import matplotlib
from config import rootDir
from itertools import islice
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from random import randint



def load_arff_data(dataset_name):
    balanceScale = rootDir + "/" + dataset_name + ".arff"

    data, meta = arff.loadarff(balanceScale)

    normalizer = Normalizer(copy=False)
    class_name = "class"

    for col in list(data.dtype.names):
        if col == "Class":
            class_name = col

    labels = data[class_name]
    labels_strings, labels_true = np.unique(labels, return_inverse=True)

    pca_data = data[[b for b in list(data.dtype.names) if b != class_name]]

    imputer = Imputer(missing_values=0, strategy="mean", axis=0)

    pca_data = pd.DataFrame(pca_data)

    le = LabelEncoder()

    column_type = []

    for col in pca_data.columns.values:
        if pca_data[col].dtypes == "object":
            le.fit(pca_data[col].values)
            pca_data[col] = le.transform(pca_data[col])
            # categorical
            column_type.append(1)
        else:
            column_type.append(0)

    pca_data = np.asarray(pca_data, dtype=np.float32)
    # clustering_data = np.asarray(clustering_data)

    pca_data = imputer.fit_transform(pca_data)
    pca_data = normalizer.fit_transform(pca_data)



    return pca_data,labels_true,labels_strings


original_data=load_arff_data("vehicle")[0]
labels=load_arff_data("vehicle")[1]
labels_strings=load_arff_data("vehicle")[2]

def take(n, iterable):

    return list(islice(iterable, n))

def plot_data(data,title):
        n_items = take(len(np.unique(labels)), matplotlib.colors.cnames.iteritems())

        colors=[name for(name,color) in n_items]
        fig = plt.figure()
        ax = plt.subplot(111)


        ax.scatter(data[:,0],data[:,1],alpha=0.5,c=labels, cmap=matplotlib.colors.ListedColormap(colors))

        handlelist = [plt.plot([], marker="o", ls="", color=color)[0] for color in colors]

        plt.legend(handlelist,np.unique(labels),loc='right',title="Classes")
        plt.xlabel("First dimention of the data set")
        plt.ylabel("Second dimention of the data set")
        plt.title("{} set".format (title))

        plt.show()
def pca(data,dimentions):

        means=np.mean(data, axis=0)
        means=means.reshape(1,means.shape[0])

        data, data_column_means = np.broadcast_arrays(data, means)
        adjusted_data=np.subtract(data,data_column_means)
        cov_mat=np.cov(adjusted_data.transpose())
        print('Covariance Matrix:\n', cov_mat)

        eigenvlaues, eigenvectors= np.linalg.eig(cov_mat)

        for idx,value in enumerate(eigenvlaues):
            print('Eigenvector {}: \n{}'.format(idx+1, eigenvectors[idx]))

            print('Eigenvalue {} from covariance matrix: {}'.format(idx+1, value))
            print(40 * '-')


        eigenvlaues=sorted(eigenvlaues,reverse=True)
        eig_pairs=[(eigenvlaues[i], eigenvectors[i]) for i in range(0, len(eigenvlaues))]

        print ("Eigenvalues in decreasing order: \n")
        for i in eig_pairs:
                print(i[0])

        components=eigenvectors[0:dimentions]
        return components,adjusted_data,means


components,adjusted_data,means= pca(original_data,3)

transformed_data=np.dot(adjusted_data,components.transpose())

reconstructed_data=np.add(adjusted_data,means)

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([transformed_data[:, 0].mean()], [transformed_data[:, 1].mean()], [transformed_data[:, 2].mean()], 'o', markersize=10, color='red', alpha=0.5)
for v in components:
    a = Arrow3D([transformed_data[:, 0].mean(), v[0]], [transformed_data[:, 0].mean(), v[1]], [transformed_data[:, 0].mean(), v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvectors')

plt.show()