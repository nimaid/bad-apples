import os
import urllib.request
from enum import Enum
import hashlib
import cv2
import numpy as np
import blend_modes as bm
import ffmpeg



# A class to represent the Bad Apple video
# Capable of multiple video qualities
# Will download and verify missing files
class BadApple:
    class Quality(Enum):
        STANDARD = 0 # 360p @ 30 FPS
        SD60FPS = 1 # 360p @ 60 FPS
        HD = 2 # 720p @ 30FPS --TODO--
        HD60FPS = 3 # 720p @ 60 FPS
        FHD = 4 # 1080p @ 30 FPS --TODO--
        FHD60FPS = 5 # 1080p @ 60 FPS --TODO--
        QHD = 6 # 1440p @ 30 FPS --TODO--
        QHD60FPS = 7 # 1440p @ 60 FPS --TODO--
        UHD = 6 # 2160p @ 30 FPS --TODO--
        UHD60FPS = 7 # 2160p @ 60 FPS --TODO--
    
    def __init__(
        self,
        quality=None
    ):
        if quality == None: # default to BadApple.Quality.STANDARD
            self.quality = self.Quality.STANDARD
        else:
            self.quality = quality
        
        self.ext = ".mp4"
        if self.quality == self.Quality.SD60FPS:
            self.name = "bad_apple@60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%4060fps.mp4"
            self.sha1 = "f6cb4b4b7c8d94dfc5edadf399e8636cd5d39082"
            self.img_scale = 1 # 360p
            self.fps_scale = 2 # 60 FPS
        elif self.quality == self.Quality.HD60FPS:
            self.name = "bad_apple@720p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p60fps.mp4"
            self.sha1 = "af382d0bb69e467ab6a3e57635c2448e5242742f"
            self.img_scale = 2 # 720p
            self.fps_scale = 2 # 60 FPS
        else: # default is also BadApple.Quality.STANDARD
            self.name = "bad_apple"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple.mp4"
            self.sha1 = "d248203e4f8a88433bee75cf9d0e746386ba4b1b"
            self.img_scale = 1 # 360p
            self.fps_scale = 1 # 30 FPS
        self.filename = self.name + self.ext
        
        # Get the file
        self.ensure_bad_apple()
        
        self.filename_full = os.path.realpath(self.filename)
        
    # Some useful return codes
    class ErrorCode(Enum):
        ERR_NONE = 0
        ERR_CONNECTION_FAILED = 1
        ERR_CONNECTION_OTHER = 2
        ERR_FILE_MISSING = 3
        ERR_FILE_CORRUPT = 4
        ERR_USER_STOP = 5

    # Function to validate if the video is missing or corrupt
    def validate_bad_apple(self):
        # Check to make sure the file is there
        if not os.path.isfile(self.filename):
            return self.ErrorCode.ERR_FILE_MISSING
        
        # Check if file is corrupted
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks
        download_sha1 = hashlib.sha1()
        with open(self.filename, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                download_sha1.update(data)
        download_sha1 = download_sha1.hexdigest()
        if self.sha1 != download_sha1:
            return self.ErrorCode.ERR_FILE_CORRUPT
        
        # All good
        return self.ErrorCode.ERR_NONE

    # Function to download a local copy of bad apple
    def download_bad_apple(self):
        # Download the video locally
        print("Downloading the video...")
        try:
            file_location, result = urllib.request.urlretrieve(self.url, self.filename)
        except urllib.error.URLError:
            # Download failed due to connection issues
            # May be temporary, or a sign the upload was removed.
            return self.ErrorCode.ERR_CONNECTION_FAILED
        except KeyboardInterrupt:
            return self.ErrorCode.ERR_USER_STOP
        except:
            # Some other connection related issues
            return self.ErrorCode.ERR_CONNECTION_OTHER
        print("Video downloaded!\n")
        
        print("Starting checksum verification...")
        result = self.validate_bad_apple()
        if result in [self.ErrorCode.ERR_FILE_MISSING, self.ErrorCode.ERR_FILE_CORRUPT]:
            print("Checksum did not match! Download may have gotten corrupted.")
            return result
        print("Checksum verified!\n")
        
        # All good
        return self.ErrorCode.ERR_NONE

    # Function to make sure a non-corrupt copy of bad apple is available locally
    # It will download one if needed
    def ensure_bad_apple(self):
        result = self.validate_bad_apple()
        while result != self.ErrorCode.ERR_NONE:
            # Retry
            print("Bad Apple not found. Trying to get Bad Apple...\n")
            retry_result = self.download_bad_apple()
            if retry_result == self.ErrorCode.ERR_USER_STOP:
                os.remove(self.filename)
                raise KeyboardInterrupt("User interrupted the download process.")
            result = self.validate_bad_apple()
        print("Bad Apple is ready!\n")


# Create the BadApple object
ba = BadApple(BadApple.Quality.STANDARD)

# Make capture object for playback
video = cv2.VideoCapture(ba.filename)
# Check that the capture object is ready
if video.isOpened():
    print('Video successfully opened!\n')
else:
    print('Something went wrong!\n')

# How much to scale outputs up by
upscale_factor = 2 # 6 to go from 360p to 4K
upscale_method = cv2.INTER_CUBIC

# Get video dimensions and FPS
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
video_size = (int(frame_width) * upscale_factor, int(frame_height) * upscale_factor)
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Make output filenames
temp_filename = ba.name + "_temp" + ba.ext
new_filename = ba.name + "_edit" + ba.ext

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
    window_size = 15 * ba.img_scale
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 1, window_size, 1, 9, 3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    prev_frame = next_frame
    
    # Smooth colors with a blur
    blur_px = 11 * ba.img_scale
    blur_sigma = 200
    smooth_frame = cv2.bilateralFilter(flow_frame, blur_px, blur_sigma, blur_sigma)
    
    # Add over last motion frame by blending with lighten
    fade_amt = 4 / ba.fps_scale
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
video_original = ffmpeg.input(ba.filename)
video_new = ffmpeg.input(temp_filename)
video_muxed = ffmpeg.output(video_original.audio, video_new.video, new_filename)
ffmpeg_result = video_muxed.run()
os.remove(temp_filename) # Delete the temp file
print("\nAdded audio!")

