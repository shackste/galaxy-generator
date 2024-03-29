from typing import Dict, Union, Any
from pathlib import Path
import re
import urllib
import glob
import os
import hashlib
import requests
import html
import uuid
import io
import tempfile

import yaml
import numpy as np

import torch


PathOrStr = Union[Path, str]


def get_config(config: str) -> Dict:
    """Loads config as dict from yaml file

    Args:
        config: path to config file

    Returns:
        Dict: loaded config
    """

    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_device() -> str:
    """Returns available torch device

    Returns:
        str: available torch device
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


_cache_dir = './cashe'
def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    _feature_detector_cache = dict()

    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]


class_groups = {
    # group : indices (assuming 0th position is id)
    0: (),
    1: (1, 2, 3),
    2: (4, 5),
    3: (6, 7),
    4: (8, 9),
    5: (10, 11, 12, 13),
    6: (14, 15),
    7: (16, 17, 18),
    8: (19, 20, 21, 22, 23, 24, 25),
    9: (26, 27, 28),
    10: (29, 30, 31),
    11: (32, 33, 34, 35, 36, 37),
}


class_groups_indices = {g: np.array(ixs)-1 for g, ixs in class_groups.items()}


hierarchy = {
    # group : parent (group, label)
    2: (1, 1),
    3: (2, 1),
    4: (2, 1),
    5: (2, 1),
    7: (1, 0),
    8: (6, 0),
    9: (2, 0),
    10: (4, 0),
    11: (4, 0),
}


def make_galaxy_labels_hierarchical(labels: torch.Tensor) -> torch.Tensor:
    """ transform groups of galaxy label probabilities to follow the hierarchical order defined in galaxy zoo
    more info here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    labels is a NxL torch tensor, where N is the batch size and L is the number of labels,
    all labels should be > 1
    the indices of label groups are listed in class_groups_indices

    Return
    ------
    hierarchical_labels : NxL torch tensor, where L is the total number of labels
    """
    shift = labels.shape[1] > 37 ## in case the id is included at 0th position, shift indices accordingly
    index = lambda i: class_groups_indices[i] + shift

    for i in range(1, 12):
        ## normalize probabilities to 1
        norm = torch.sum(labels[:, index(i)], dim=1, keepdims=True)
        norm[norm == 0] += 1e-4  ## add small number to prevent NaNs dividing by zero, yet keep track of gradient
        labels[:, index(i)] /= norm
        ## renormalize according to hierarchical structure
        if i not in [1, 6]:
            parent_group_label = labels[:, index(hierarchy[i][0])]
            labels[:, index(i)] *= parent_group_label[:, hierarchy[i][1]].unsqueeze(-1)
    return labels
