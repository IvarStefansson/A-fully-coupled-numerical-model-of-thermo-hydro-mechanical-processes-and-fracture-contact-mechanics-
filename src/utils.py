import pickle
from pathlib import Path
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def write_pickle(obj, path):
    """ Write any object to pickle

    Parameters
    ----------
    obj : any object
    path : path-like
        path to storage file.
    """

    path = Path(path)
    # path.mkdir(parents=True, exist_ok=False)  # Don't overwrite existing files
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw)


def read_pickle(path):
    """ Read a stored object
    """
    with open(path, "rb") as f:
        raw = f.read()
    return pickle.loads(raw)


def write_fracture_data_txt(setup):
    """
    Write data summarizing the results to csv files. Four files are written:
        time steps
        Newton iterations for all time steps    
        normal displacement jumps on all fractures for all time steps
        tangential displacement jumps on all fractures for all time steps
    """
    folder = setup.folder_name
    if not os.path.exists(folder):
        os.makedirs(folder)

    t = np.atleast_2d(setup.export_times).T
    iterations = np.atleast_2d(setup.iterations).T

    t = t[-iterations.size :]
    fn = folder + "/"
    data = np.hstack(
        (
            t,
            iterations,
            setup.u_jumps_tangential,
            setup.u_jumps_normal,
            setup.force_tangential,
            setup.force_normal,
        )
    )
    if hasattr(setup, "well_p"):
        data = np.hstack((data, setup.well_p, setup.well_T))
    np.savetxt(fn + "data_" + setup.file_name + ".txt", data)


def prepare_plotting(**kwargs):
    plt.close("all")
    matplotlib.use("agg", warn=False, force=True)  # force non-GUI backend.

    # Number of floating points
    mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-4, 4))

    # Plotting

    # Preparing plot
    sns.set_context("paper")  # set scale and size of figures
    sns.set_palette("tab10", 10)
    itertools.cycle(sns.color_palette())  # iterate if > 10 colors are needed
    figsize = kwargs.get("figsize", (6, 8))
    fig = plt.figure(8, constrained_layout=True, figsize=figsize)
    widths = kwargs.get("widths", [5, 5, 1])
    heights = kwargs.get("heights", [1, 3, 3, 3, 3, 3, 3])
    gs = fig.add_gridspec(
        nrows=len(heights),
        ncols=len(widths),
        width_ratios=widths,
        height_ratios=heights,
    )
    return fig, gs


model_labels = [r"$M_0$", r"$M_1$", r"$M_2$"]

colors_rgb = ["#b90000", "#00af00", "#0000b9"]
