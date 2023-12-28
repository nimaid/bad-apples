import os
import urllib.parse
import urllib.request
from enum import Enum
import logging
import hashlib

from mflowm.files import VideoReader


def validate(filename, sha1):
    # Check to make sure the file is there
    if not os.path.isfile(filename):
        return False

    # Check if file is corrupted
    buffer_size = 65536  # Let's read stuff in 64kb chunks
    download_sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            download_sha1.update(data)

    download_sha1 = download_sha1.hexdigest()
    if sha1 != download_sha1:
        return False

    return True


class Quality(Enum):
    SD = 0  # 360p @ 30 FPS
    SD60 = 1  # 360p @ 60 FPS
    HD = 2  # 720p @ 30FPS
    HD60 = 3  # 720p @ 60 FPS
    FHD = 4  # 1080p @ 30 FPS
    FHD60 = 5  # 1080p @ 60 FPS
    QHD = 6  # 1440p @ 30 FPS
    QHD60 = 7  # 1440p @ 60 FPS
    UHD = 8  # 2160p @ 30 FPS
    UHD60 = 9  # 2160p @ 60 FPS


# Function to download a local copy of bad apple
def download_bad_apple(quality, path="."):
    match quality:
        case Quality.SD:
            url = "https://archive.org/download/bad-apple-resources/bad_apple.mp4"
            sha1 = "d248203e4f8a88433bee75cf9d0e746386ba4b1b"
        case Quality.SD60:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%4060fps.mp4"
            sha1 = "f154318c4049b665aa4fa4dc819b10c2c34ff97e"
        case Quality.HD:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p.mp4"
            sha1 = "333bae3a21b4e514e06f5a6b1104dfb0c698411e"
        case Quality.HD60:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p60fps.mp4"
            sha1 = "15c22498e6abf3fb0f7ca73d39d281a3e5c0c706"
        case Quality.FHD:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p.mp4"
            sha1 = "b8fef140406312d4bc2a51936d7de9c47fe02e8b"
        case Quality.FHD60:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p60fps.mp4"
            sha1 = "549491981229b937dc5f3987851d343a456828f2"
        case Quality.QHD:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p.mp4"
            sha1 = "012425b863987ef84e4aafabbb66998dd6e15d51"
        case Quality.QHD60:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p60fps.mp4"
            sha1 = "6204b3173ec745f4c583b6dde11f858a7886b8d0"
        case Quality.UHD:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p.mp4"
            sha1 = "028ec64b3c909a92b6532b32a2473f735667feb0"
        case Quality.UHD60:
            url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p60fps.mp4"
            sha1 = "d5dcaef680abbff71c0e9fb9de130d45a4ba2cb7"
        case _:
            raise ValueError("An invalid quality setting was provided")

    filename = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
    filename = os.path.realpath(os.path.join(path, filename))

    if not os.path.exists(filename):
        # Download the video locally
        logging.info("Downloading video... ({})".format(filename))
        try:
            file_location, result = urllib.request.urlretrieve(url, filename)
        except KeyboardInterrupt:
            os.remove(filename)
            raise KeyboardInterrupt("User interrupted the download process.")
        logging.info("Video downloaded!")

    logging.info("Starting checksum verification...")
    is_valid = validate(filename=filename, sha1=sha1)
    if not is_valid:
        logging.fatal("Checksum invalid!")
        os.remove(filename)
        return None
    logging.info("Checksum verified.")

    return filename


# A class to represent the Bad Apple video
# Capable of multiple video qualities
# Will download and verify missing files
class BadApple(VideoReader):
    def __init__(
            self,
            quality=None
    ):
        if quality is None:  # default to SD
            quality = Quality.SD

        # Get the file
        filename = download_bad_apple(quality=quality)
        if not filename:
            raise Exception("Failed to download Bad Apple in the desired quality")

        # Init the parent class
        super().__init__(filename=filename)
