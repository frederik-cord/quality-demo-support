import numpy as np
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import logging
from pandas import DataFrame

from matplotlib.patches import Rectangle

from .data_grabber import DataGrabber
from .torch import StatsLog

logger = logging.getLogger()


def plot_dataframe_samples(
    grabber: DataGrabber,
    dataframe: DataFrame,
    file_path: Optional[Union[Path, str]] = None,
):
    """
    Will plot the first 9 samples of the dataframe.
    If the dataunit corresponding to the data_hash and the object_hash has a bounging
    box, this will also be displayed.
    If file_path specified, the figure will be stored at that path.
    :param grabber: The datagrabber to fetch images.
    :param dataframe: The dataframe with the data unit information needed (dh, oh)
    :param file_path: Optional file_path to store figure.
    """
    rows, cols = np.round(np.sqrt(len(dataframe)))
    fig, ax = plt.subplots(rows, cols, figsize=(4*rows, 3*cols))
    ax = ax.reshape(-1)

    for a, row in zip(ax, dataframe.to_dict(orient="records")):
        img = grabber.image_from_hash(**row)
        obj = grabber.object_from_hashes(**row)

        a.imshow(img)
        a.axis("off")
        a.set_title(f"name: {row['name']}, pred iou: {row['iou_estimates_min']:.2f}")

        if hasattr(obj, "bounding_box"):
            bbox = obj.bounding_box
            h, w, c = img.shape
            y = bbox.y * h
            x = bbox.x * w
            h_i = bbox.h * h
            w_i = bbox.w * w
            p = Rectangle((x, y), w_i, h_i, color="magenta", alpha=0.3)
            a.add_patch(p)

    fig.tight_layout()
    fig.patch.set_facecolor("white")

    if file_path:
        logger.info(f"Saving figure to {file_path}")
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        plt.savefig(file_path)


def make_plots_from_logs(*logs: StatsLog, file_path=None):
    """
    :param logs: list of StatsLog objects.
    :param file_path: path to save figure
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    colors = {}

    # == Train loss == #
    a = ax[0, 0]
    a.grid()
    a.set_title("train loss")
    for log in logs:
        p = a.plot(log.train_loss_list, label=log.name)
        a.set_xlabel("epoch")
        a.set_ylabel("train loss")
        colors[log] = p[0].get_color()
    a.legend()

    def plot_attr(a, attr_name: str):
        y_label = " ".join(attr_name.split("_")[:-1])
        a.set_title(y_label)
        a.grid()
        for log in logs:
            p = a.plot(getattr(log, attr_name), color=colors[log])
            a.set_xlabel("epoch")
            a.set_ylabel(y_label)

    # == Train acc == #
    plot_attr(ax[0, 1], "train_acc_list")

    # == Val loss == #
    plot_attr(ax[1, 0], "val_loss_list")

    # == Val acc == #
    plot_attr(ax[1, 1], "val_acc_list")
    fig.tight_layout()
    fig.patch.set_facecolor("white")

    if file_path:
        logger.info(f"Saving figure to {file_path}")
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)
        plt.savefig(file_path)

    return fig
