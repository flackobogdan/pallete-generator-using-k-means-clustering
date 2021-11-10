from sklearn.cluster import KMeans
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt

default_pallete = 5
downscale_size = 200

def get_palette(img, num_clusers=default_pallete):
    kmeans = KMeans(n_clusters=num_clusers)
    channels = img.shape[2]
    points = img.reshape(-1, channels)
    kmeans.fit(points)
    return kmeans.cluster_centers_

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        dest='palette_size',
        type=int,
        help=f"pallete size (default={default_pallete})",
        default=default_pallete,
    )
    parser.add_argument("file", help="path to the image file")

    args = parser.parse_args()

    img = cv2.imread(args.file)
    cv2.imshow('image', img)

    num_clusers = args.palette_size

    #downsample
    img = cv2.resize(img, (downscale_size, downscale_size))
    #cv2 cita boje kao BGR umesto RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = get_palette(img, num_clusers=num_clusers)
    #array[None] daje dimenziju
    #npr. colors.shape (num_clusters, 3) -> colors.shape (1, num_clusters, 3)
    #jer pyplot ocekuje matricu
    #i scale na [0-1]
    colors = colors[None]/255.

    plt.figure(figsize=(3*num_clusers, 3))
    plt.imshow(colors)
    plt.show()