import logging
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Any, Dict, Optional
from PIL import Image, ImageOps
import numpy as np

from cord.client import CordClient
from cord_pytorch_dataset.dataset import get_label_row, extract_frames
from cord_pytorch_dataset.objects import (
    Ontology,
    ImageLabelRow,
    VideoLabelRow,
    DataUnit,
    DataUnitObject,
)
from cord_pytorch_dataset.utils import download_file
from concurrent.futures import ThreadPoolExecutor as Executor

logger = logging.getLogger(__name__)


def get_data_unit_image(data_unit: DataUnit, cache_dir: Path) -> Optional[Path]:
    """
    Fetches image either from cache dir or by downloading and caching image. By default, only the image path will
    be returned as a Path object.
    Args:
        data_unit: The data unit that specifies what image to fetch.
        cache_dir: The directory to fetch cached results from, and to cache results to.
    Returns: The image as a Path.
    """
    is_video = "video" in data_unit.data_type
    if is_video:
        video_hash, frame_idx = data_unit.data_hash.split("_")
        video_dir = cache_dir / "videos"
        video_file = f"{video_hash}.{data_unit.extension}"
        img_dir = video_dir / video_hash
        img_file = f"{frame_idx}.jpg"

        os.makedirs(video_dir, exist_ok=True)
    else:
        img_dir = cache_dir / "images"
        img_file = f"{data_unit.data_hash}.{data_unit.extension}"

    full_img_pth = img_dir / img_file
    if not full_img_pth.exists():
        # Check that there is a link
        assert (
            data_unit.data_link[:8] == "https://"
        ), f"`data_unit.data_link` not downloadable.\n{data_unit.data_link}"

        if is_video:
            # Extract frames images
            if not os.path.exists(video_dir / video_file):
                logger.info(f"Downloading video {video_file}")
                download_file(
                    data_unit.data_link, video_dir, fname=video_file, progress=None
                )
            extract_frames(video_dir / video_file, img_dir)
        else:
            logger.debug(f"Downloading image {full_img_pth}")
            download_file(data_unit.data_link, img_dir, fname=img_file, progress=None)

    return full_img_pth


class DataGrabber:
    def __init__(
        self,
        project_id: str,
        api_key: str,
        cache_dir: str = "/tmp/cord_data",
        download: bool = True,
    ):
        self.download = download
        self.cache_dir = Path(cache_dir)
        self.client = CordClient.initialise(project_id, api_key)

        project = self.client.get_project()

        self.title = project.get("title", "unknown")
        self.did_warn = False

        label_rows = project.get("label_rows")
        label_rows = [lr for lr in label_rows if lr["label_hash"] is not None]

        self.ontology = Ontology(**project.get("editor_ontology"))
        os.makedirs(self.cache_dir / "labels", exist_ok=True)
        os.makedirs(self.cache_dir / "images", exist_ok=True)
        os.makedirs(self.cache_dir / "videos", exist_ok=True)

        self.data_units: List[DataUnit] = []
        logger.info("Preparing data units")
        for lr in tqdm(label_rows, desc="Processing label rows."):
            full_lr = get_label_row(
                lr.get("label_hash"),
                self.client,
                self.cache_dir,
                download=self.download,
                force=False,
            )

            # Skip if label row is empty (download == False and lr not cached)
            if not full_lr:
                continue

            if full_lr.get("data_type").lower() == "video":
                lr = VideoLabelRow(self.ontology, **full_lr)
            else:
                lr = ImageLabelRow(self.ontology, **full_lr)
            self.data_units.extend(lr.data_units)

        # get_data_unit_image(du, self.cache_dir, download=self.download, force=False)

        self.objects: Dict[str, Dict[str, Any]] = {}
        for i, du in enumerate(self.data_units):
            if len(du.objects) == 0:
                continue

            self.objects[du.data_hash] = {"__idx__": i, "__du__": du}
            for j, obj in enumerate(du.objects):
                self.objects[du.data_hash][obj.object_answer.object_hash] = {
                    "idx": j,
                    "file": None,
                    "obj": obj,
                }
        self.images = {}

    def cache_all_videos(self, num_workers=8):
        lrs = set()
        for du in self.data_units:
            if isinstance(du.label_row, VideoLabelRow):
                lrs.add(du.label_row)
        du_to_prepare = [lr.data_units[0] for lr in lrs]

        from functools import partial

        _extract_video = partial(get_data_unit_image, cache_dir=self.cache_dir)

        logger.info("Caching all videos.")
        with Executor(max_workers=num_workers) as exe:
            _ = list(
                tqdm(
                    exe.map(_extract_video, du_to_prepare),
                    total=len(du_to_prepare),
                    desc="Extracting frames from videos",
                )
            )
        logger.info("Done caching videos.")

    def image_from_hash(self, data_hash, **kwargs):
        # IMAGE FILE
        if data_hash not in self.objects:
            raise ValueError("Unknown data_hash")
        data = self.objects[data_hash]
        data_unit = data["__du__"]

        img_file = self.images.get(data_hash, None)
        if img_file is None:
            img_file = get_data_unit_image(data_unit, self.cache_dir)
            self.images["data_hash"] = img_file

        logger.debug(f"Loading image: {img_file}")
        img = Image.open(img_file)
        try:
            img = ImageOps.exif_transpose(img)

        except AttributeError as e:
            if not self.did_warn:
                logger.warning(
                    "Image may appear flipped and off from bounding box, "
                    "because `exif_transpose` didn't work."
                )
            self.did_warn = True

        return np.array(img)

    def object_from_hashes(self, data_hash, object_hash, **kwargs) -> DataUnitObject:
        # CHECK DATA UNIT
        if data_hash not in self.objects:
            raise ValueError("Unknown data_hash")
        data = self.objects[data_hash]

        # OBJECT
        if object_hash not in data:
            raise ValueError(f"Unknown object hash for data_hash {data_hash}")

        object_dict = data[object_hash]
        return object_dict["obj"]
