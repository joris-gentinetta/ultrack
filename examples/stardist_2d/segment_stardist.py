# stardist / tensorflow env variables setup
import os
from os.path import join

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from stardist.models import StarDist2D
from tifffile import imread
from pathlib import Path

import numpy as np
from segment_utils import normalize, array_apply

# data_dir = Path("/cluster/scratch/jorisg/data")
#data is one folder up:
# data_dir = Path("../data") #better:
data_dir = join(Path(__file__).parent.parent, "data")
img_path = join(data_dir, "test.tif")

n_frames = 2

imgs = imread(img_path)
imgs = imgs[:, 1:, :, :]

if n_frames is not None:
    imgs = imgs[:n_frames]
imgs = np.swapaxes(imgs, 1, 3)

image = np.mean(imgs, axis=-1)

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

# store np array: stardist_labels:
os.makedirs(data_dir, exist_ok=True)
np.save(join(data_dir, 'stardist_labels.npy'), stardist_labels)
