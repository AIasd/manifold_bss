import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Plot results in 2D
def plot_samples(S):
    # colorize = dict(c=S[:, 0], cmap=plt.cm.get_cmap('rainbow', 2))
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                alpha=0.5)
    plt.hlines(0, -2, 2)
    plt.vlines(0, -2, 2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
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
    z = np.ones((x.shape))  # (x/2+y/2)**2
    XS = np.vstack((x, y, z)).T
    return XS


def tSNE_exp(num, distance, perp):
    d = distance/2

    # Generate sample data
    S1x = np.random.uniform(-0.5-d, -d, size=(num, 1))
    S1y = np.random.uniform(-1, 1, size=(num, 1))
    S1 = np.hstack((S1x, S1y))

    S2x = np.random.uniform(d, 0.5+d, size=(num, 1))
    S2y = np.random.uniform(-1, 1, size=(num, 1))
    S2 = np.hstack((S2x, S2y))

    S = np.vstack((S1, S2))

    # Create 2D manifold in 3D
    XS = make_manifold(S)

    # Decrease dimension via t-SNE
    model_tsne = TSNE(n_components=2, perplexity=perp)
    X_tsne = model_tsne.fit_transform(XS)

    return S, XS, X_tsne

if __name__ == "__main__":

    # Specify number of points in each cluster, distance
    # between the two cluster and perplexity of each run

    num = 500
    d = [0.2, 0.05, 0.0001]
    perp = [10, 30]
    col = len(d)
    row = len(perp)

    fig = plt.figure()

    for i in range(col):
        for j in range(row):
            S, XS, X_tsne = tSNE_exp(num, d[i], perp[j])
            if j == 0:
                fig.add_subplot(row+1, col, i+1)
                plot_samples(S)
                plt.title('a='+str(d[i]))
            fig.add_subplot(row+1, col, i+1+(j+1)*col)
            plot_samples(X_tsne / np.std(X_tsne))
            plt.title('sigma='+str(perp[j]))

    plt.show()
