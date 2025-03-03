from typing import Callable, List

import napari
import numpy as np
import pytest

from ultrack.widgets import LabelsToContoursWidget

pytest.skip(reason="DEPRECATED", allow_module_level=True)


def test_labels_to_contours_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    multiple_labels: List[np.ndarray],  # noqa: F811
) -> None:

    viewer = make_napari_viewer()
    for label in multiple_labels:
        viewer.add_labels(label)

    viewer.layers.selection.update(viewer.layers)

    assert len(viewer.layers.selection) == len(viewer.layers)

    widget = LabelsToContoursWidget(viewer)
    widget._run_btn.clicked.emit()

    assert "contours" in viewer.layers

    foreground = viewer.layers["foreground"].data

    for lb in multiple_labels:
        for t in range(foreground.shape[0]):
            mask = lb[t] > 0
            assert np.all(foreground[t][mask] > 0)
