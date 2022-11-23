from skimage.color import rgb2lab, rgb2xyz, xyz2lab, lab2rgb
from skimage.io import imread, imshow
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple


# class Color(NamedTuple):
#     rgb: Tuple[int, int, int]
#     lab: Tuple[float, float, float]


def float2int(x):
    return (255 * x).round().astype(np.uint8)


def extract_colors(img: Image.Image, n_colors=8, downsample_size=128):
    assert img.mode == ("RGB")

    rgb = img.convert("RGB").resize((downsample_size, downsample_size))
    rgb = np.array(img)
    lab = rgb2lab(rgb)
    X = lab.reshape(-1, 3)

    kms = KMeans(n_clusters=n_colors, n_init=3, random_state=9)
    kms.fit(X)

    lab_centers = kms.cluster_centers_
    rgb_centers = float2int(lab2rgb(lab_centers))
    return dict(rgb=rgb_centers, lab=lab_centers)


if __name__ == "__main__":

    img = Image.open("ref2.jpg")
    centroids = extract_colors(img)

    img.resize([256, 256])
    _ = imshow(centroids["rgb"][None, ...])
