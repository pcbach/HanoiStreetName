import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import math
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
import matplotlib.image as mpimg
import sys

plt.style.use("ggplot")


# 2nd-polynomial surface
def func(X, A, B, C, D, E, F):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    x, y = X

    return (
        # x**3
        # + I * y**3
        # + H * x**2 * y
        # + G * y**2 * x
        +(A * x**2)
        + (B * y**2)
        + (C * x * y)
        + (D * x)
        + (E * y)
        + F
    )


def flatFunc(Z):
    return np.log(1 + Z)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# read by default 1st sheet of an excel file
df = pd.read_excel("XLPlotInput.xlsx")
fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
# print(df)

year = []
for ind in df.index:
    if math.isnan(df["Year"][ind]):
        df = df.drop([ind])
        continue
    year.append(df["Year"][ind])
df = df.reset_index(drop=True)
# print(df)
year = list(set(year))
year.sort()
# print(year)
# print(len(year))
points3D = np.zeros((len(df), 3))

levels = 49
plotMask = True
plotBar = False
plotBG = True
plotSurface = False
plotContour = False
plotColorBar = False
zAxisMax = 0.5
if sys.argv[2] == "draft":
    dwSmpl = 5
else:
    dwSmpl = 1

color = cm.jet(np.linspace(0, 1, levels + 1))

for ind in df.index:
    for j in range(levels + 1):
        idx = int(j * len(year) / levels)
        # print(ind, idx) ->
        if (
            df["Year"][ind] <= year[min(idx, len(year) - 1)]
            and df["Year"][ind] <= int(sys.argv[4])
            and df["Year"][ind] >= int(sys.argv[3])
        ):
            # print(j, df["Year"][ind], ind, year[min(idx, len(year) - 1)], idx)
            if plotBar == True:
                ax1.bar3d(
                    df["X"][ind],
                    df["Y"][ind],
                    0,
                    0.003,
                    0.003,
                    flatFunc((j + 1) / levels) / flatFunc(1) * zAxisMax,
                    color=color[j],
                )
            else:
                ax1.scatter(
                    df["X"][ind],
                    df["Y"][ind],
                    zAxisMax * 0.1,
                    # flatFunc((j + 1) / levels) / flatFunc(1) * zAxisMax,
                    marker="o",
                    s=2,
                    color=color[j],
                    edgecolors="black",
                    lw=0.1,
                )

            points3D[ind, :] = [
                df["X"][ind],
                df["Y"][ind],
                flatFunc((j + 1) / levels) / flatFunc(1) * zAxisMax,
            ]
            break


minLat = 20.9614
maxLat = 21.0956
minLon = 105.7764
maxLon = 105.9415
origin = (21.0285, 105.85895)
scaling = 7.451564828614016

minLatS = (minLat - origin[0]) * scaling
maxLatS = (maxLat - origin[0]) * scaling
minLonS = (minLon - origin[1]) * scaling
maxLonS = (maxLon - origin[1]) * scaling
# print(minLatS, maxLatS, minLonS, maxLonS)

if plotMask:
    img = mpimg.imread("MapXLHighResMask.png")
    print(img.shape)
    stepY = (maxLatS - minLatS) / (img.shape[0])
    stepX = (maxLonS - minLonS) / (img.shape[1])
    Y1 = np.arange(maxLatS, minLatS, -stepY)
    X1 = np.arange(minLonS, maxLonS, stepX)
    X1, Y1 = np.meshgrid(X1, Y1)
    # print(Y1.shape)
    ax1.plot_surface(
        X1,
        Y1,
        np.atleast_2d(zAxisMax),
        rstride=dwSmpl,
        cstride=dwSmpl,
        facecolors=img,
        shade=False,
    )
if plotBG:
    img = mpimg.imread("MapXLHighRes.png")
    print(img.shape)
    stepY = (maxLatS - minLatS) / (img.shape[0])
    stepX = (maxLonS - minLonS) / (img.shape[1])
    Y1 = np.arange(maxLatS, minLatS, -stepY)
    X1 = np.arange(minLonS, maxLonS, stepX)
    X1, Y1 = np.meshgrid(X1, Y1)
    # print(Y1.shape)
    ax1.plot_surface(
        X1,
        Y1,
        np.atleast_2d(0),
        rstride=dwSmpl,
        cstride=dwSmpl,
        facecolors=img,
        shade=False,
    )
if plotContour or plotSurface:
    parameters, covariance = curve_fit(
        func, [points3D[:, 0], points3D[:, 1]], points3D[:, 2] + 0.001
    )
    # print(parameters, covariance)
    stepY = (maxLatS - minLatS) / 100
    stepX = (maxLonS - minLonS) / 100
    Y = np.arange(maxLatS, minLatS, -stepY)
    X = np.arange(minLonS, maxLonS, stepX)
    # create coordinate arrays for vectorized evaluations
    X, Y = np.meshgrid(X, Y)
    # calculate Z coordinate array
    Z = func(np.array([X, Y]), *parameters)
    if plotSurface:
        ax1.plot_surface(
            X,
            Y,
            flatFunc(Z) / flatFunc(1) * zAxisMax,
            cmap=plt.cm.jet,
            linewidth=0,
            antialiased=True,
            alpha=0.3,
            shade=False,
        )

    if plotContour:
        ax1.contour(
            X,
            Y,
            flatFunc(Z) / flatFunc(1) * zAxisMax,
            int(levels),
            offset=zAxisMax * 0.1,
            cmap=plt.cm.jet,
            linewidths=0.5,
        )

if plotColorBar:
    m = cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(np.linspace(0, 1, 5))
    cbar = plt.colorbar(m, ax=plt.gca(), shrink=0.4)
    # m = cm.ScalarMappable(cmap=plt.cm.jet)
    # m.set_array(flatFunc(Z) / flatFunc(1) * zAxisMax)
    # cbar = plt.colorbar(m, ax=plt.gca(), shrink=0.5)
    # cbar.set_alpha(0.3)
    level = np.arange(0, levels + 1, 1)
    idx = np.int64(level * (len(year) - 1) / (levels))
    # print(ind, idx)
    # print(np.array(year)[idx])
    cbar.set_ticks(np.linspace(0, 1, len(idx)))
    cbar.set_ticklabels(np.array(year)[idx])
    cbar.ax.tick_params(width=0.1, labelsize=2)
ax1.set_axis_off()
fig.set_facecolor("white")
ax1.set_facecolor("white")

ax1.set_zlim([0, zAxisMax])
set_axes_equal(ax1)
ax1.view_init(elev=90, azim=-90)

plt.axis("off")
# plt.tight_layout()
# plt.show()
if sys.argv[2] == "draft":
    dpi = 300
else:
    dpi = 1000
print("Saving")
plt.savefig(
    "Output/plot" + sys.argv[1] + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0
)
