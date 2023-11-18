
# stardist / tensorflow env variables setup
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

import napari
import numpy as np
from napari.utils.notebook_display import nbscreenshot
from tqdm import tqdm
from rich.pretty import pprint

from stardist.models import StarDist2D

from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.imgproc import normalize
from ultrack.utils import estimate_parameters_from_labels, labels_to_edges
from ultrack.utils.array import array_apply
from ultrack.config import MainConfig
from tifffile import imread, imwrite

print('imports done')
img_path = Path("/cluster/scratch/jorisg/data/test.tif")


n_frames = 2

imgs = imread(img_path)
imgs = imgs[:, 1:, :, :]


if n_frames is not None:
    imgs = imgs[:n_frames]
imgs = np.swapaxes(imgs, 1, 3)

viewer = napari.Viewer(show=False)
viewer.window.resize(1800, 1000)
layers = viewer.add_image(imgs, channel_axis=3, name="raw")

image=np.mean([viewer.layers[i].data for i in [0, 1, 2]], axis=0)


model = StarDist2D.from_pretrained("2D_versatile_fluo")
stardist_labels = np.zeros_like(image, dtype=np.int32)
def predict(frame: np.ndarray, model: StarDist2D) -> np.ndarray:
    """Normalizes and computes stardist prediction."""
    frame = normalize(frame, gamma=1.0)
    labels, _ = model.predict_instances_big(
        frame, "YX", block_size=560, min_overlap=96, show_progress=False,
    )
    return labels

array_apply(
        image,
    out_array=stardist_labels,
        func=predict,
    model=model,
    )

viewer.add_labels(stardist_labels, name="stardist")


nbscreenshot(viewer)


# The `labels_to_edges` converts labels into Ultrack's expected input, a detection and edges maps (cells' boundaries). The `sigma` parameter blurs the cell boundaries, assisting the segmentation hypothesis estimation, the goal is to make the boundaries similar to a distance transform inside the cells.

detection, edges = labels_to_edges(stardist_labels, sigma=4.0)  # multiple labels can be used with [labels_0, labels_1, ...]



viewer.add_image(detection, visible=False)
viewer.add_image(edges, blending="additive", colormap="magma")
nbscreenshot(viewer)


# ## 2. Tracking
# 
# Now that we have our `detection` and `edges` you can call the `track` function for the tracking on the contour representation of the cells.
# 
# The `track` function is composed of three steps that can also be called individually:
# - `segment`: Computes the segmentation hypothesis for tracking;
# - `link`: Links and assign edge weights to the segmentation hypothesis;
# - `solve`: Solves the tracking problem by selecting the strongly connected segmentation hypothesis.
# 
# Each of these steps requires its own configuration, which we'll set up below. Its documentation can be found [here](https://github.com/royerlab/ultrack/blob/main/ultrack/config/README.md).
# 
# We create our configuration instance and print its default values.




config = MainConfig()
pprint(config)


# To assist setting the parameters we inspect Stardist's results using the function `estimate_params_from_labels` from `ultrack.utils`.
# The `min_area` was selected to eliminate a few small segments which could be noise or incorrect segmentations.
# For the `max_area` we also avoid the right tail of the distribution because it could also be outliers.

params_df = estimate_parameters_from_labels(stardist_labels, is_timelapse=True)
params_df["area"].plot(kind="hist", bins=100, title="Area histogram")




config.segmentation_config.min_area = 50
config.segmentation_config.max_area = 950
config.segmentation_config.n_workers = 8


# The remaining parameters are harder to estimate without ground-truth data, hence they were tuned by trial and error.
# From our experience setting the `power` parameter to 3 or 4 yields better results, specially in challenging scenarios. Note that, you must adjust the other `*_weight` accordingly when `power` is updated.


config.linking_config.max_distance = 25
config.linking_config.n_workers = 8

config.tracking_config.appear_weight = -1
config.tracking_config.disappear_weight = -1
config.tracking_config.division_weight = -0.1
config.tracking_config.power = 4
config.tracking_config.bias = -0.001
config.tracking_config.solution_gap = 0.0

pprint(config)


track(
    detection=detection,
    edges=edges,
    config=config,
    overwrite=True,
)


tracks_df, graph = to_tracks_layer(config)
labels = tracks_to_zarr(config, tracks_df)



viewer.add_tracks(tracks_df[["track_id", "t", "y", "x"]].values, graph=graph)
viewer.add_labels(labels)

viewer.layers["stardist"].visible = False
viewer.layers["edges"].visible = False

nbscreenshot(viewer)

