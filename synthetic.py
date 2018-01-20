import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap, TSNE


# Plot results in 2D
def plot_samples(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('y')


def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)


# Make a 3D manifold from a set of 2D points
def make_manifold(X):
    x = X[:, 0]
    y = X[:, 1]
    z = (x/2+y/2)**2
    return np.vstack((x, y, z)).T


if __name__ == "__main__":
    # Generate sample data
    rng = np.random.RandomState(42)
    # S = rng.standard_t(1.5, size=(5000, 2))
    S = rng.uniform(-2, 2, size=(3000, 2))

    # Mix data
    # A = np.array([[2, 3], [2, 1]])  # Mixing matrix
    # X = np.dot(S, A.T)  # Generate observations
    X = rotate(S, 45)

    # Create 2D manifold in 3D
    XS = make_manifold(X)

    # Decrease dimension via Isomap/t-SNE
    model_tsne = TSNE(n_components=2)
    model_isomap = Isomap(n_neighbors=15, n_components=2, eigen_solver='dense')
    X_tsne = model_tsne.fit_transform(XS)
    X_isomap = model_isomap.fit_transform(XS)

    # Rotate axes via ICA
    ica = FastICA(random_state=rng)
    # Estimate the sources
    S_isomap = ica.fit(X_isomap).transform(X_isomap)
    S_tsne = ica.fit(X_tsne).transform(X_tsne)

    # #############################################################################

    fig = plt.figure()
    fig.add_subplot(3, 3, 1)
    plot_samples(S / S.std())
    plt.title('True Independent Sources')

    fig.add_subplot(3, 3, 2)
    plot_samples(X / np.std(X))
    plt.title('Observations')

    ax = fig.add_subplot(3, 3, 3, projection='3d')
    # colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
    ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2])
    plt.title('Manifold')

    fig.add_subplot(3, 3, 4)
    plot_samples(X_isomap / np.std(X_isomap))
    plt.title('Recovered(Isomap)')

    fig.add_subplot(3, 3, 5)
    plot_samples(S_isomap / S_isomap.std(axis=0))
    plt.title('Recovered(Isomap+ICA)')

    fig.add_subplot(3, 3, 6)
    plot_samples(X_tsne / np.std(X_tsne))
    plt.title('Recovered(TSNE)')

    fig.add_subplot(3, 3, 7)
    plot_samples(S_tsne / S_tsne.std(axis=0))
    plt.title('Recovered(TSNE+ICA)')

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.show()
