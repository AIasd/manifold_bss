import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE


# Plot results in 2D
def plot_samples(S, tot_num):
    colorize = dict(c=S[:, 0], cmap=plt.cm.get_cmap('rainbow', tot_num))
    plt.scatter(S[:, 0], S[:, 1], **colorize, s=1, marker='o', zorder=10,
                alpha=0.5)
    plt.hlines(0, -2, 2)
    plt.vlines(0, -2, 2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')


def plot_samples_3d(XS, ax, tot_num):
    colorize = dict(c=XS[:, 0], cmap=plt.cm.get_cmap('rainbow', tot_num))
    ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize, s=2)


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


# size e.g. (4, 3) will generate 4 horizontal rectangular blocks
# at three different heights
def tSNE_exp(size, num, d, perp):
    h_num = size[0]
    v_num = size[1]

    S_list = []

    # Generate sample data
    for i in range(h_num):
        for j in range(v_num):
            Sx = np.random.uniform((1+d)*i, (1+d)*i+1, size=(num, 1))
            Sy = np.random.uniform(-1, 1, size=(num, 1))
            Sz = np.random.uniform((1+d)*j, (1+d)*j+1, size=(num, 1))
            S1 = np.hstack((Sx, Sy, Sz))
            S_list.append(S1)
    XS = np.vstack(S_list)

    # Decrease dimension via t-SNE
    model_tsne = TSNE(n_components=2, perplexity=perp)
    X_tsne = model_tsne.fit_transform(XS)

    return XS, X_tsne

if __name__ == "__main__":

    # Specify number of points in each cluster, distance
    # between the two cluster and perplexity of each run

    num = 700
    d = [0.2, 0.05, 0.0001]
    perp = [10, 30]
    col = len(d)
    row = len(perp)

    h_num = 3
    v_num = 3
    tot_num = h_num*v_num
    size = [h_num, v_num]

    fig = plt.figure()

    for i in range(col):
        for j in range(row):
            XS, X_tsne = tSNE_exp(size, num, d[i], perp[j])
            if j == 0:
                ax = fig.add_subplot(row+1, col, i+1, projection='3d')
                plot_samples_3d(XS, ax, tot_num)
                plt.title('a='+str(d[i]))
            fig.add_subplot(row+1, col, i+1+(j+1)*col)
            plot_samples(X_tsne / np.std(X_tsne), tot_num)
            plt.title('perplexity='+str(perp[j]))

    plt.show()
