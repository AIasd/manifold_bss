import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import offsetbox

from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.decomposition import FastICA
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_mldata


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


# Generate 2D visualization of the MNIST dataset using model and model+ICA
def mnist_run(model):
    mnist = fetch_mldata('MNIST original')

    # use only 1/20 of the data: full dataset takes a long time!
    data = mnist.data[::20]
    target = mnist.target[::20]

    # Show the embedding of all the training data points in 2D using model
    proj = model.fit_transform(data)

    f0 = plt.figure()
    f0.add_subplot(111)
    plt.scatter(proj[:, 0], proj[:, 1], c=target, cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)

    # Comparison of applying model and model+ICA
    # Extract subset consists of "1"
    data = mnist.data[mnist.target == 1][::2]
    # Apply model
    proj = model.fit_transform(data)
    # Apply ICA
    ica = FastICA(n_components=2)
    proj_ica = ica.fit_transform(proj)

    f1 = plt.figure()
    f1.add_subplot(111)
    plot_components(data,
                    proj=proj,
                    images=data.reshape((-1, 28, 28)))
    plt.title('model')

    f2 = plt.figure()
    f2.add_subplot(111)
    plot_components(data,
                    proj=proj_ica,
                    images=data.reshape((-1, 28, 28)))
    plt.title('model+ICA')
    plt.show()


if __name__ == "__main__":
    # FACES
    face_run(Isomap(n_neighbors=5, n_components=2, eigen_solver='dense'))

    # MNIST
    # mnist_run(Isomap(n_neighbors=5, n_components=2, eigen_solver='dense'))
