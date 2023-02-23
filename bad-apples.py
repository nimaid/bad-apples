import os
import urllib.request
from enum import Enum
import hashlib
import cv2
import numpy as np
import blend_modes as bm
import ffmpeg

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
    ERR_USER_STOP = 5

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
    except KeyboardInterrupt:
        return ErrorCode.ERR_USER_STOP
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
        if retry_result == ErrorCode.ERR_USER_STOP:
            os.remove(bad_apple_video)
            raise KeyboardInterrupt("User interrupted the download process.")
        result = validate_bad_apple()
    print("Bad Apple is ready!\n")


# Make sure we have the file before we go on
ensure_bad_apple()

# Make capture object for playback
video = cv2.VideoCapture(bad_apple_video)
# Check that the capture object is ready
if video.isOpened():
    print('Video successfully opened!\n')
else:
    print('Something went wrong!\n')

# How much to scale outputs up by
upscale_factor = 1 # 6 to go from 360p to 4K
upscale_method = cv2.INTER_CUBIC

# Get video dimensions and FPS
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
video_size = (int(frame_width) * upscale_factor, int(frame_height) * upscale_factor)
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Make output filenames
ba_name, ba_ext = os.path.splitext(bad_apple_video)
temp_filename = ba_name + "_temp" + ba_ext
new_filename = ba_name + "_edit" + ba_ext

# Delete existing one
try:
    os.remove(temp_filename)
    os.remove(new_filename)
except:
    pass

# Start writing new file
new_video = cv2.VideoWriter(
    filename=temp_filename,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=fps,
    frameSize=video_size
)

# Make playback window
windowName = 'Bad Apple'
cv2.namedWindow(windowName)
# Read the first frame
ret, frame1 = video.read()
prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prev_motion_frame = np.zeros_like(frame1)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
# Play the video
user_stopped = False
frame_count = 1
while True:
    ret, frame2 = video.read() # Read a single frame 
    if not ret: # This mean it could not read the frame 
         print("Could not read the frame, video is likely over.")   
         cv2.destroyWindow(windowName)
         video.release()
         break
    
    frame_count += 1
    
    print("Processing frame {}/{}".format(frame_count, total_frames))
    
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Get flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 1, 15, 1, 9, 3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    prev_frame = next_frame
    
    # Smooth colors with a blur
    blur_px = 11
    blur_sigma = 200
    smooth_frame = cv2.bilateralFilter(flow_frame, blur_px, blur_sigma, blur_sigma)
    
    # Add over last motion frame by blending with lighten
    fade_amt = 2
    img_sub = np.ones_like(prev_motion_frame) * fade_amt
    # Darken last motion frame
    motion_frame_bg = np.subtract(prev_motion_frame, img_sub.astype(np.int16)).clip(0, 255).astype(np.uint8)
    # Do the lighten
    motion_frame = np.clip(np.maximum(motion_frame_bg, smooth_frame), 0, 256).astype(np.uint8)
    
    
    # Blend motion colors over original video
    next_frame_bgr = cv2.cvtColor(next_frame, cv2.COLOR_GRAY2BGR)
    # Add alpha channel for blend_modes module
    next_frame_bgr_alpha = cv2.cvtColor(next_frame_bgr, cv2.COLOR_RGB2RGBA)
    motion_frame_alpha = cv2.cvtColor(motion_frame, cv2.COLOR_RGB2RGBA)
    # Convert to float for blend_modes module
    next_frame_bgr_float = next_frame_bgr_alpha / 255.0 
    motion_frame_float = motion_frame_alpha / 255.0 
    # Do the blending
    blend_frame = bm.difference(next_frame_bgr_float, motion_frame_float, 1.0)
    #blend_frame = motion_frame_float
    # Convert back to int
    final_frame_alpha = (blend_frame * 255).astype(np.uint8)
    # Strip alpha channel
    final_frame = cv2.cvtColor(final_frame_alpha, cv2.COLOR_RGBA2RGB)
    
    
    # Display frame
    cv2.imshow(windowName, final_frame)
    
    # Scale frame for outputs
    if upscale_factor != 1:
        final_video_frame = cv2.resize(final_frame, video_size, 0, 0, interpolation = upscale_method)
    else:
        final_video_frame = final_frame
    
    # Save frame
    new_video.write(final_video_frame)
    
    # Update last motion frame
    prev_motion_frame = motion_frame
    
    # Exit hotkey
    stop_playing = False
    waitKey = (cv2.waitKey(1) & 0xFF)
    if waitKey == ord('q'): # If Q pressed
        stop_playing = True
    
    if stop_playing:
        print("Closing video and exiting...")
        user_stopped = True
        cv2.destroyWindow(windowName)
        video.release()
        break

# Save new video
new_video.release()


# If user quit, stop here
if user_stopped:
    exit()


# Mux original audio and new video together (lossless)
print("\nAdding audio...\n")
video_original = ffmpeg.input(bad_apple_video)
video_new = ffmpeg.input(temp_filename)
video_muxed = ffmpeg.output(video_original.audio, video_new.video, new_filename)
ffmpeg_result = video_muxed.run()
os.remove(temp_filename) # Delete the temp file
print("\nAdded audio!")

