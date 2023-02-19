import os
import urllib.request
from enum import Enum
import hashlib

bad_apple_video = "bad_apple.mp4"

# The current best source for the original video from NicoNico
bad_apple_url = "https://archive.org/download/nicovideo-sm8628149/nicovideo-sm8628149_4c8a655c13612a596d6b97c58797d3c622adebddc6436264e47e615fdccb9d21.mp4"
bad_apple_sha1 = "d248203e4f8a88433bee75cf9d0e746386ba4b1b"

# Some useful return codes
class ErrorCode(Enum):
    ERR_NONE = 0
    ERR_CONNECTION_FAILED = 1
    ERR_CONNECTION_OTHER = 2
    ERR_FILE_MISSING = 3
    ERR_FILE_CORRUPT = 4

# Function to validate if the video is missing or corrupt
def validate_bad_apple():
    filename = bad_apple_video
    # Check to make sure the file is there
    if not os.path.isfile(filename):
        return ErrorCode.ERR_FILE_MISSING
    
    # Check if file is corrupted
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks
    download_sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            download_sha1.update(data)
    download_sha1 = download_sha1.hexdigest()
    if bad_apple_sha1 != download_sha1:
        return ErrorCode.ERR_FILE_CORRUPT
    
    # All good
    return ErrorCode.ERR_NONE

# Function to download a local copy of bad apple
def download_bad_apple():
    # Download the video locally
    print("Downloading the video...")
    filename = bad_apple_video
    try:
        file_location, result = urllib.request.urlretrieve(bad_apple_url, filename)
    except urllib.error.URLError:
        # Download failed due to connection issues
        # May be temporary, or a sign the upload was removed.
        return ErrorCode.ERR_CONNECTION_FAILED
    except:
        # Some other connection related issues
        return ErrorCode.ERR_CONNECTION_OTHER
    print("Video downloaded!\n")
    
    print("Starting checksum verification...")
    result = validate_bad_apple()
    if result in [ErrorCode.ERR_FILE_MISSING, ErrorCode.ERR_FILE_CORRUPT]:
        print("Checksum did not match! Download may have gotten corrupted.")
        return result
    print("Checksum verified!\n")
    
    # All good
    return ErrorCode.ERR_NONE

# Function to make sure a non-corrupt copy of bad apple is available locally
# It will download one if needed
def ensure_bad_apple():
    result = validate_bad_apple()
    while result != ErrorCode.ERR_NONE:
        # Retry
        print("Bad Apple not found. Trying to get Bad Apple...\n")
        retry_result = download_bad_apple()
        result = validate_bad_apple()
    print("Bad Apple is ready!\n")


# Make sure we have the file before we go on
ensure_bad_apple()

