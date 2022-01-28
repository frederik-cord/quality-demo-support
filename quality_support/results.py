import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

from api_client import ApiClient

logger = logging.getLogger(__name__)


@dataclass
class KeyResults:
    key: str
    df: pd.DataFrame
    scores: dict


def download_and_load_inference_results(
    api_client: ApiClient, refresh: bool = False
) -> Dict[str, KeyResults]:
    """
    Will fetch all the inference results from the api and load them into dataframes.
    The result
    :param api_client:
    :param refresh:
    :return: dictionary of results. Each key represent a mode of errors, e.g., label
        errors and bounding box errors.
    """
    # Results url
    data = api_client.get_inference_urls()

    root = Path(os.getcwd())
    for key in data.keys():
        name, *_ = key.split(".")
        target_path = root / name

        if target_path.is_dir() and not refresh:
            logger.info(
                f"Results already exists in {target_path}. "
                f"To refresh set get_results(project_id, refresh=True)"
            )
            continue
        elif not target_path.is_dir() or refresh:
            logger.info(f"Downloading and extracting {key} to {target_path}")
            url = data[key]["url"]
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download
                    target_path = Path(temp_dir) / key
                    with open(target_path, "wb") as f:
                        f.write(response.raw.read())

                    # Extract
                    with tarfile.open(target_path, "r:gz") as tar:
                        tar.extractall(path=root / name)

    keys = [k.split(".")[0] for k in data.keys()]
    results = {k: _get_results_from_key(k) for k in keys}

    logger.info("Found the following inference results:")
    for res in results.values():
        logger.info(f" {res.key} columns:  {res.df.columns.to_list()}")
        logger.info(f" {res.key} scores:  {res.scores}")

    return results


def _get_results_from_key(key) -> KeyResults:
    dataframes = []
    scores = {}

    root = Path(os.getcwd())
    for k, v in np.load(root / key / "results.npz").items():
        if v.ndim == 0:
            # Single value, wont fit into df
            scores[k] = v.item()
            continue
        if v.ndim == 1:
            # Single column
            cols = [k]
        elif v.ndim == 2:
            # Multiple columns
            cols = [f"{k}_{i}" for i in range(v.shape[1])]
            cols += [f"{k}_min", f"{k}_max"]

            v = np.concatenate(
                [v, v.min(axis=-1, keepdims=True), v.max(axis=-1, keepdims=True)],
                axis=-1,
            )
            logger.debug(f"{k} shape: {v.shape}")
        else:
            raise ValueError("Don't know how to interpret 3D arrays.")

        # === DATAFRAME === #
        dataframes.append(
            pd.DataFrame(data=v, columns=cols, index=np.arange(v.shape[0]))
        )

    # === METADATA === #
    with open(root / key / "metadata.json") as f:
        metadata = json.load(f)
    dataframes.append(pd.DataFrame(data=metadata, index=np.arange(len(metadata))))

    return KeyResults(key, pd.concat(dataframes, axis=1, join="inner"), scores)


def download_file(
    url: str,
    destination: Path,
    fname: str,
    byte_size=1024,
):
    local_filename = destination / fname
    if local_filename.is_file():
        return local_filename

    r = requests.get(url, stream=True)

    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=byte_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return local_filename
