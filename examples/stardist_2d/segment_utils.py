# stardist / tensorflow env variables setup

import itertools
from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm


def normalize(
        image: ArrayLike,
        gamma: float,
        lower_q: float = 0.001,
        upper_q: float = 0.9999,
) -> ArrayLike:
    """
    Normalize image to between [0, 1] and applies a gamma transform (x ^ gamma).

    Parameters
    ----------
    image : ArrayLike
        Images as an T,Y,X,C array.
    gamma : float
        Expoent of gamma transform.
    lower_q : float, optional
        Lower quantile for normalization.
    upper_q : float, optional
        Upper quantile for normalization.

    Returns
    -------
    ArrayLike
        Normalized array.
    """
    frame = image
    frame = frame - np.quantile(frame, lower_q)
    frame = frame / np.quantile(frame, upper_q)
    frame = np.clip(frame, 0, 1)

    if gamma != 1.0:
        frame = np.power(frame, gamma)

    return frame


def array_apply(
        *in_arrays: ArrayLike,
        out_array: ArrayLike,
        func: Callable,
        axis: Union[Tuple[int], int] = 0,
        **kwargs,
) -> None:
    """Apply a function over a given dimension of an array.

    Parameters
    ----------
    in_arrays : ArrayLike
        Arrays to apply function to.
    out_array : ArrayLike
        Array to store result of function.
    func : function
        Function to apply over time.
    axis : Union[Tuple[int], int], optional
        Axis of data to apply func, by default 0.
    args : tuple
        Positional arguments to pass to func.
    **kwargs :
        Keyword arguments to pass to func.
    """
    name = func.__name__ if hasattr(func, "__name__") else type(func).__name__

    for arr in in_arrays:
        if arr.shape != out_array.shape:
            raise ValueError(
                f"Input arrays {arr.shape} must have the same shape as the output array {out_array.shape}."
            )

    if isinstance(axis, int):
        axis = (axis,)

    stub_slicing = [slice(None) for _ in range(out_array.ndim)]
    multi_indices = list(itertools.product(*[range(out_array.shape[i]) for i in axis]))
    for indices in tqdm(multi_indices, f"Applying {name} ..."):
        for a, i in zip(axis, indices):
            stub_slicing[a] = i
        indexing = tuple(stub_slicing)
        out_array[indexing] = func(*[a[indexing] for a in in_arrays], **kwargs)
