import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap, TSNE


# Plot results in 2D
def plot_samples(S, axis_list=None):
    colorize = dict(c=S[:, 0], cmap=plt.cm.get_cmap('rainbow', 2))
    plt.scatter(S[:, 0], S[:, 1], **colorize, s=2, marker='o', zorder=10,
                alpha=0.5)
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


def tSNE_exp(num, distance):
    d = distance/2

    # Generate sample data
    rng = np.random.RandomState(42)

    S1x = rng.uniform(-2, -d, size=(num, 1))
    S1y = rng.uniform(-2, 2, size=(num, 1))
    S1 = np.hstack((S1x, S1y))

    S2x = rng.uniform(d, 2, size=(num, 1))
    S2y = rng.uniform(-2, 2, size=(num, 1))
    S2 = np.hstack((S2x, S2y))

    S = np.vstack((S1, S2))

    # Create 2D manifold in 3D
    XS = make_manifold(S)

    # Decrease dimension via t-SNE
    model_tsne = TSNE(n_components=2, perplexity=10.0)
    X_tsne = model_tsne.fit_transform(XS)

    return S, XS, X_tsne

if __name__ == "__main__":

    num1 = 80
    num2 = 1000

    S, XS, X_tsne = tSNE_exp(num1, 0.3)
    S2, XS2, X_tsne2 = tSNE_exp(num2, 0.3)

    # #############################################################################

    fig = plt.figure()

    fig.add_subplot(2, 3, 1)
    plot_samples(S / S.std())
    plt.title('True Independent Sources '+str(num1))

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    colorize = dict(c=S[:, 0], cmap=plt.cm.get_cmap('rainbow', 2))
    ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)
    plt.title('Manifold '+str(num1))

    fig.add_subplot(2, 3, 3)
    plot_samples(X_tsne / np.std(X_tsne))
    plt.title('Recovered(TSNE) '+str(num1))

    fig.add_subplot(2, 3, 4)
    plot_samples(S2 / S2.std())
    plt.title('True Independent Sources '+str(num2))

    ax2 = fig.add_subplot(2, 3, 5, projection='3d')
    colorize = dict(c=S2[:, 0], cmap=plt.cm.get_cmap('rainbow', 2))
    ax2.scatter3D(XS2[:, 0], XS2[:, 1], XS2[:, 2], **colorize)
    plt.title('Manifold '+str(num2))

    fig.add_subplot(2, 3, 6)
    plot_samples(X_tsne2 / np.std(X_tsne2))
    plt.title('Recovered(TSNE) '+str(num2))

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.show()
