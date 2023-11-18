# stardist / tensorflow env variables setup
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pickle
import napari
import numpy as np
import pandas as pd

from tifffile import imread

# (T, Y, X, C) data, where T=time, Y, X =s patial coordinates and C=channels
img_path = Path("../data/test.tif")

# optional, useful for a quick look
# for all frames `n_frames = None`
n_frames = 2

imgs = imread(img_path)
imgs = imgs[:, 1:, :, :]

# chunks = (1, *imgs.shape[1:-1], 1) # chunk size used to compress data

if n_frames is not None:
    imgs = imgs[:n_frames]
imgs = np.swapaxes(imgs, 1, 3)

# dataset_path = Path("Fluo-N2DL-HeLa/01")


viewer = napari.Viewer()
viewer.window.resize(1800, 1000)
# viewer.open(sorted(dataset_path.glob("*.tif")), stack=True)
layers = viewer.add_image(imgs, channel_axis=3, name="raw")

stardist_labels = np.load('data/stardist_labels.npy')
viewer.add_labels(stardist_labels, name="stardist")

detection = np.load('data/detection.npy')
edges = np.load('data/edges.npy')
viewer.add_image(detection, visible=False)
viewer.add_image(edges, blending="additive", colormap="magma")

tracks_df = pd.read_pickle('data/tracks.pkl')
with open('data/graph.pkl', 'rb') as f:
    graph = pickle.load(f)
labels = np.load('data/labels.npy')
viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph)
viewer.add_labels(labels)

viewer.layers["stardist"].visible = False
viewer.layers["edges"].visible = False
