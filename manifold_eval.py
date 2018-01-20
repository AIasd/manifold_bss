import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from mpl_toolkits import mplot3d
from matplotlib import offsetbox

from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.decomposition import FastICA
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_mldata

from time import time
from random import randint


def plot_components(data, proj, images=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = plt.gca()
    ax.plot(proj[:, 0], proj[:, 1], '.k')

    if images is not None:
        min_dist_2 = 2*(thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)


# Generate 2D visualization of the face dataset using model and model+ICA
def face_run(model):
    faces = fetch_lfw_people(min_faces_per_person=30)
    # Apply model
    proj = model.fit_transform(faces.data)
    # Apply ICA
    ica = FastICA(n_components=2)
    proj_ica = ica.fit_transform(proj)

    f1 = plt.figure()
    f1.add_subplot(111)
    plot_components(faces.data,
                    proj=proj,
                    images=faces.images[:, ::2, ::2])
    plt.title('model')

    f2 = plt.figure()
    f2.add_subplot(111)
    plot_components(faces.data,
                    proj=proj_ica,
                    images=faces.images[:, ::2, ::2])
    plt.title('model+ICA')
    plt.show()


# Get the indices of the knn
def indices_of_knn(point_dist, k=5):
    return sorted(range(len(point_dist)), key=lambda i: point_dist[i])[:k]


def get_dist_matrix(data):
    num_data = data.shape[0]
    dist = [[0 for _ in range(num_data)] for _ in range(num_data)]
    # Calculate Euclidean distances of any two pairs in both high-d and low-d
    for i in range(num_data):
        for j in range(i+1, num_data):
            dist[i][j] = la.norm(data[i] - data[j])
            dist[j][i] = dist_highd[i][j]
    return dist


# Get the knn precision of the low dimensional mapping.
# Note: The indices of the points in dist_highd and dist_lowd
# must correspond to each other.
# k: the number of nearest neighbors.
# n: the number of data points.
# dist_highd: the original data (which usually has high dimensions).
# dist_lowd: the data mapped to low dimensions.
def knn_precision(dist_highd, dist_lowd, k=5, n=100):
    # For each data point, get the indices of the knn in both high-d
    # and low-d, and take the average of the precision
    precision_avg = 0
    for i in range(n):
        knn_highd = indices_of_knn(dist_highd[i], k)
        knn_lowd = indices_of_knn(dist_lowd[i], k)
        # Record the number of corresponding neighbor indices
        c = 0
        for i in knn_highd:
            if i in knn_lowd:
                c += 1
        # Add the precision of the current data point
        # to the averafe precision
        precision_avg += 1/n * c/k
    return precision_avg

def dijkstra(data_highd, dist_highd, start_ind, end_ind):
    return []

def get_line(start, end):
    return (0, 0)

# Get the sampled curve error of the low dimension mapping

def curve_error(data_highd, data_lowd, dist_highd, dist_lowd, n=10):
    num_data = data_highd.shape[0]
    for i in range(n):
        start_ind = randint(0, num_data)
        end_ind = randint(0, num_data)
        path_inds = dijkstra(data_highd, dist_highd, start_ind, end_ind)
        k, b = get_line(data_lowd[start_ind], data_lowd[end_ind])

if __name__ == "__main__":
    faces = fetch_lfw_people(min_faces_per_person=30)
    model_isomap = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
    model_tsne = TSNE(n_components=2)
    model_lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2, eigen_solver='dense')

    # Apply models
    proj_isomap = model_isomap.fit_transform(faces.data)
    proj_tsne = model_tsne.fit_transform(faces.data)
    proj_lle = model_lle.fit_transform(faces.data)

    # Get distance matrices
    dist_highd = get_dist_matrix(faces.data)
    dist_isomap_lowd = get_dist_matrix(faces.data)
    dist_tsne_lowd = get_dist_matrix(faces.data)
    dist_lle_lowd = get_dist_matrix(faces.data)

    # Calculate knn precisions for each embedding
    print(knn_precision(dist_highd, dist_isomap_lowd, 5, 500))
    print(knn_precision(dist_highd, dist_tsne_lowd, 5, 500))
    print(knn_precision(dist_highd, dist_lle_lowd, 5, 500))
