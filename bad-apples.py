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
        SD = 0 # 360p @ 30 FPS
        SD60 = 1 # 360p @ 60 FPS
        HD = 2 # 720p @ 30FPS
        HD60 = 3 # 720p @ 60 FPS
        FHD = 4 # 1080p @ 30 FPS
        FHD60 = 5 # 1080p @ 60 FPS
        QHD = 6 # 1440p @ 30 FPS
        QHD60 = 7 # 1440p @ 60 FPS
        UHD = 8 # 2160p @ 30 FPS
        UHD60 = 9 # 2160p @ 60 FPS
    
    def __init__(
        self,
        quality=None
    ):
        if quality is None: # default to BadApple.Quality.SD
            self.quality = self.Quality.SD
        else:
            self.quality = quality
        
        self.ext = ".mp4"
        if self.quality == self.Quality.SD:
            self.name = "bad_apple"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple.mp4"
            self.sha1 = "d248203e4f8a88433bee75cf9d0e746386ba4b1b"
            self.img_scale = 1 # 360p
            self.fps_scale = 1 # 30 FPS
        elif self.quality == self.Quality.SD60:
            self.name = "bad_apple@60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%4060fps.mp4"
            self.sha1 = "f154318c4049b665aa4fa4dc819b10c2c34ff97e"
            self.img_scale = 1 # 360p
            self.fps_scale = 2 # 60 FPS
        elif self.quality == self.Quality.HD:
            self.name = "bad_apple@720p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p.mp4"
            self.sha1 = "333bae3a21b4e514e06f5a6b1104dfb0c698411e"
            self.img_scale = 2 # 720p
            self.fps_scale = 1 # 30 FPS
        elif self.quality == self.Quality.HD60:
            self.name = "bad_apple@720p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p60fps.mp4"
            self.sha1 = "15c22498e6abf3fb0f7ca73d39d281a3e5c0c706"
            self.img_scale = 2 # 720p
            self.fps_scale = 2 # 60 FPS
        elif self.quality == self.Quality.FHD:
            self.name = "bad_apple@1080p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p.mp4"
            self.sha1 = "b8fef140406312d4bc2a51936d7de9c47fe02e8b"
            self.img_scale = 3 # 1080p
            self.fps_scale = 1 # 30 FPS
        elif self.quality == self.Quality.FHD60:
            self.name = "bad_apple@1080p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p60fps.mp4"
            self.sha1 = "549491981229b937dc5f3987851d343a456828f2"
            self.img_scale = 3 # 1080p
            self.fps_scale = 2 # 60 FPS
        elif self.quality == self.Quality.QHD:
            self.name = "bad_apple@1440p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p.mp4"
            self.sha1 = "012425b863987ef84e4aafabbb66998dd6e15d51"
            self.img_scale = 4 # 1440p
            self.fps_scale = 1 # 30 FPS
        elif self.quality == self.Quality.QHD60:
            self.name = "bad_apple@1440p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p60fps.mp4"
            self.sha1 = "6204b3173ec745f4c583b6dde11f858a7886b8d0"
            self.img_scale = 4 # 1440p
            self.fps_scale = 2 # 60 FPS
        elif self.quality == self.Quality.UHD:
            self.name = "bad_apple@2160p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p.mp4"
            self.sha1 = "028ec64b3c909a92b6532b32a2473f735667feb0"
            self.img_scale = 6 # 2160p
            self.fps_scale = 1 # 30 FPS
        elif self.quality == self.Quality.UHD60:
            self.name = "bad_apple@2160p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p60fps.mp4"
            self.sha1 = "d5dcaef680abbff71c0e9fb9de130d45a4ba2cb7"
            self.img_scale = 6 # 2160p
            self.fps_scale = 2 # 60 FPS
        else:
            raise ValueError("An invalid quality setting was provided to the BadApple class!")
        self.filename = self.name + self.ext
        self.width = 480 * self.img_scale
        self.height = 360 * self.img_scale
        self.size = (self.width, self.height)
        self.fps = 30.0 * self.fps_scale
        self.shape = (self.height, self.width, 3)
        
        # Get the file
        self.ensure_bad_apple()
        
        self.filename_full = os.path.realpath(self.filename)
        
        # Open the video file
        self.video = None
        result = self.open()
        if result != self.ErrorCode.ERR_NONE:
            raise RuntimeError("Could not open source video {}".format(self.filename))
        # Get total frames
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame = None
        self.frame_num = 0
    
    # The function called when the object is deleted
    def __del__(self):
        self.close()
        
    # Some useful return codes
    class ErrorCode(Enum):
        ERR_NONE = 0
        ERR_CONNECTION_FAILED = 1
        ERR_CONNECTION_OTHER = 2
        ERR_FILE_MISSING = 3
        ERR_FILE_CORRUPT = 4
        ERR_USER_STOP = 5
        ERR_CANT_OPEN = 6

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
        print("Downloading video... ({})".format(self.filename))
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
            print("Bad Apple video not found in the desired quality. Trying to download it...\n")
            retry_result = self.download_bad_apple()
            if retry_result == self.ErrorCode.ERR_USER_STOP:
                os.remove(self.filename)
                raise KeyboardInterrupt("User interrupted the download process.")
            result = self.validate_bad_apple()
        print("Bad Apple is ready! ({})\n".format(self.filename))
    
    # Function to open the source video
    def open(self):
        # Check if already opened
        if self.video is not None:
            return self.ErrorCode.ERR_NONE
        # Make capture object for playback
        video = cv2.VideoCapture(self.filename)
        # Check that the capture object is ready
        if not video.isOpened():
            print("Could not open source video!\n")
            self.video = None
            return self.ErrorCode.ERR_CANT_OPEN
        else:
            self.video = video
            return self.ErrorCode.ERR_NONE
    
    # Function to close the source video
    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
    
    # Function to read next frame
    def read_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.frame_num += 1
            self.frame = frame
            return frame
        else: # Cant read frame, video is probably over
            self.frame_num = -1
            self.frame = None
            return None


# A class to handle single-pass motion flow computation
class AppleMotionFlow:
    def __init__(
        self,
        bad_apple, # Existing BadApple object
        flow_window=15, # Relative to video scale, higher gets bigger motions but is blurrier
        flow_layers=4, # Number of layers in computation, more is better but slower
        flow_iterations=4, # Number of iterations per layer, more is better but slower
        flow_poly_n=7,
        flow_poly_sigma=1.5,
        blur_amount=1.0, # Relative to video scale and flow window
        blur_sigma=2.0,
        fade_speed=4 # Relative to FPS
    ):
        self.ba = bad_apple
        self.flow_window_size = flow_window * self.ba.img_scale
        self.flow_layers = flow_layers
        self.flow_iterations = flow_iterations
        self.flow_poly_n = flow_poly_n
        self.flow_poly_sigma = flow_poly_sigma
        self.blur_px = max(round(blur_amount * self.flow_window_size), 1)
        # Make sure it's odd
        if self.blur_px%2 == 0:
            self.blur_px += 1
        self.blur_sigma = blur_sigma
        self.fade_amt = max(round(fade_speed / ba.fps_scale), 1)
        
        # Make image to fade with
        self.img_sub = np.ones(ba.shape) * self.fade_amt
        
        # Make HSV array
        self.hsv = np.zeros(ba.shape).astype(np.uint8)
        self.hsv[..., 1] = 255 # Full saturation
        
        # Init frames
        self.frame = None
        self.src_frame = None
        self.prev_src_frame = None
        self.motion_frame = None
        self.prev_motion_frame = None
        
        # Init other vars
        self.flow = None
    
    # Function to compute a single flow frame from 2 input frames
    # Does not modify object at all
    def get_flow(
        self,
        first_frame,
        second_frame
    ):  
        # Convert to gray
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
        
        # Get flow
        if self.flow is None or True: # Oh god it SUCKS why doesn't it work never give it the flow for the love of god
            flow_in = None
            flow_opts = 0
        else:
            flow_in = self.flow
            flow_opts = cv2.OPTFLOW_USE_INITIAL_FLOW
        flow = cv2.calcOpticalFlowFarneback(
            first_frame_gray,
            second_frame_gray,
            flow_in,
            pyr_scale=0.5, # Layer "pyramid" size ratio
            levels=self.flow_layers,
            winsize=self.flow_window_size,
            iterations=self.flow_iterations,
            poly_n=self.flow_poly_n,
            poly_sigma=self.flow_poly_sigma,
            flags=flow_opts
        )
        self.flow = flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang*180/np.pi/2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_frame = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
    
        # Smooth colors with a blur
        '''
        smooth_frame = cv2.bilateralFilter(
            flow_frame,
            self.blur_px,
            self.blur_sigma,
            self.blur_sigma
        )
        '''
        smooth_frame = cv2.GaussianBlur(flow_frame, (self.blur_px,self.blur_px), self.blur_sigma)
        return smooth_frame
    
    # Function to read next frame from video file
    def get_next_src_frame(self):
        frame = self.ba.read_frame()
        # Check if frame could not be gotten
        if frame is None:
            return False
        self.prev_src_frame = self.src_frame
        self.src_frame = frame
        return True
    # Alternate way to set with external frames
    def set_next_src_frames(self, src_frame, prev_src_frame=None):
        if prev_src_frame is None:
            self.prev_src_frame = self.src_frame
        else:
            self.prev_src_frame = prev_src_frame
        self.src_frame = src_frame
    
    # Computes the next motion flow frame
    def calc_motion_frame(
        self,
        read_new_frame=True, # Set to False if the self.ba object gets updated with read_frame() elsewhere
        trails=True # If we want to use trails or just get a single motion frame
    ):
        if read_new_frame:
            result = self.get_next_src_frame()
            # If we haven't gotten the second frame yet, get it
            if self.prev_src_frame is None:
                result = not ((not result) | (not self.get_next_src_frame()))
        elif (self.src_frame is None) or (self.prev_src_frame is None):
            raise ValueError("Function was told not to read new source frames, but needs new source frames.")
        else:
            result = True
        
        if not result:
            self.motion_frame = None
            return None
        
        # Compute the optical flow motion frame
        motion_frame = self.get_flow(self.prev_src_frame, self.src_frame)
        
        # If we are going to do layering
        if trails:
            # layer over old motion frame if it exists
            if self.prev_motion_frame is None: # We don't have both, just set the motion current frame be the first frame
                layered_motion_frame = motion_frame
            else:
                # Darken last motion frame
                motion_frame_bg = np.subtract(self.motion_frame, self.img_sub.astype(np.int16)).clip(0, 255).astype(np.uint8)
                # Add over last motion frame by blending with lighten
                layered_motion_frame = np.clip(np.maximum(motion_frame_bg, motion_frame), 0, 256).astype(np.uint8)
        else:
            layered_motion_frame = motion_frame
            
            
        self.prev_motion_frame = self.motion_frame
        self.motion_frame = layered_motion_frame
        
        return self.motion_frame
    
    # Function used to layer motion colors over the source video
    # Does not modify the object
    # Currently using difference
    def layer_over_image(self, bottom, top):
        # Add alpha channel for blend_modes module
        bottom_alpha = cv2.cvtColor(bottom, cv2.COLOR_RGB2RGBA)
        top_alpha = cv2.cvtColor(top, cv2.COLOR_RGB2RGBA)
        
        # Convert to float for blend_modes module
        bottom_alpha_float = bottom_alpha / 255.0 
        top_alpha_float = top_alpha / 255.0 
        
        # Do the blending
        blend_frame = bm.difference(top_alpha_float, bottom_alpha_float, 1.0)
        
        # Convert back to int
        final_frame_alpha = (blend_frame * 255).astype(np.uint8)
        
        # Strip alpha channel
        final_frame = cv2.cvtColor(final_frame_alpha, cv2.COLOR_RGBA2RGB)
        
        return final_frame
    
    def calc_full_frame(
        self,
        read_new_frame=True # Set to false if the self.ba object gets updated with read_frame() elsewhere
    ):
        motion_frame = self.calc_motion_frame(read_new_frame)
        if motion_frame is None:
            self.frame = None
            return None
        layered_frame = self.layer_over_image(motion_frame, self.src_frame)
        
        self.frame = layered_frame
        return self.frame

# A class to handle multi-pass motion flow computation
class AppleMotionFlowMulti:
    def __init__(
        self,
        bad_apple, # Existing BadApple object
        flow_windows_count=3, # Number of flow calculations to do at different sizes
        flow_windows_min=5, # Relative to video scale, higher gets bigger motions but is blurrier
        flow_windows_max=25, # Relative to video scale, higher gets bigger motions but is blurrier
        flow_windows_balance=True, # If we want to divide each motion layer brightness based on number of windows
        flow_layers=4, # Number of layers in computation, more is better but slower
        flow_iterations=4, # Number of iterations per layer, more is better but slower
        flow_poly_n=7,
        flow_poly_sigma=1.5,
        blur_amount=1.0, # Relative to video scale and flow window
        blur_sigma=2.0,
        fade_speed=4 # Relative to FPS
    ):
        self.num_windows = flow_windows_count
        self.flow_windows_balance = flow_windows_balance
        # Make AppleMotionFlow objects
        flow_windows = [int(round(x)) for x in np.linspace(flow_windows_min, flow_windows_max, self.num_windows)]
        self.ba = bad_apple
        self.mf = [AppleMotionFlow(
            bad_apple=self.ba,
            flow_window=flow_window,
            flow_layers=flow_layers,
            flow_iterations=flow_iterations,
            flow_poly_n=flow_poly_n,
            flow_poly_sigma=flow_poly_sigma,
            blur_amount=blur_amount,
            blur_sigma=blur_sigma,
            fade_speed=fade_speed # Relative to FPS
        ) for flow_window in flow_windows]
        
        # Init frames
        self.frame = None
        self.src_frame = None
        self.prev_src_frame = None
        self.motion_frame = None
        self.prev_motion_frame = None
    
    # Function to read next frame from video file
    def get_next_src_frame(self):
        frame = self.ba.read_frame()
        # Check if frame could not be gotten
        if frame is None:
            return False
        self.prev_src_frame = self.src_frame
        self.src_frame = frame
        return True
    # Alternate way to set with external frames
    def set_next_src_frames(self, src_frame, prev_src_frame=None):
        if prev_src_frame is None:
            self.prev_src_frame = self.src_frame
        else:
            self.prev_src_frame = prev_src_frame
        self.src_frame = src_frame
    
    # Function used to layer motion frames together
    # Does not modify the object
    # Currently using lighten
    def layer_motion_frames(self, bottom, top):
        # Add alpha channel for blend_modes module
        bottom_alpha = cv2.cvtColor(bottom, cv2.COLOR_RGB2RGBA)
        top_alpha = cv2.cvtColor(top, cv2.COLOR_RGB2RGBA)
        
        # Convert to float for blend_modes module
        bottom_alpha_float = bottom_alpha / 255.0 
        top_alpha_float = top_alpha / 255.0 
        
        # Do the blending
        blend_frame = bm.lighten_only(top_alpha_float, bottom_alpha_float, 1.0)
        
        # Convert back to int
        final_frame_alpha = (blend_frame * 255).astype(np.uint8)
        
        # Strip alpha channel
        final_frame = cv2.cvtColor(final_frame_alpha, cv2.COLOR_RGBA2RGB)
        
        return final_frame
    
    # Computes the next motion flow frame
    def calc_motion_frame(self):
        result = self.get_next_src_frame()
        # If we haven't gotten the second frame yet, get it
        if self.prev_src_frame is None:
            result = not ((not result) | (not self.get_next_src_frame()))
        
        if not result:
            self.motion_frame = None
            return None
        
        # Compute the optical flow motion frame
        motion_frame = np.zeros(self.ba.shape).astype(np.uint8)
        for motion_flow in self.mf:
            motion_flow.set_next_src_frames(self.src_frame, self.prev_src_frame)
            motion_flow_frame = motion_flow.calc_motion_frame(read_new_frame=False, trails=False)
            # Darken based on number of windows
            if self.flow_windows_balance:
                motion_flow_frame = np.around(motion_flow_frame / self.num_windows).astype(np.uint8)
            # Layer over previous motion flow frames
            motion_frame = self.layer_motion_frames(motion_flow_frame, motion_frame)
        
        # layer over old motion frame if it exists
        if self.prev_motion_frame is None: # We don't have both, just set the motion current frame be the first frame
            layered_motion_frame = motion_frame
        else:
            # Darken last motion frame
            motion_frame_bg = np.subtract(self.motion_frame, self.mf[0].img_sub.astype(np.int16)).clip(0, 255).astype(np.uint8)
            # Add over last motion frame by blending with lighten
            layered_motion_frame = np.clip(np.maximum(motion_frame_bg, motion_frame), 0, 256).astype(np.uint8)
            
            
        self.prev_motion_frame = self.motion_frame
        self.motion_frame = layered_motion_frame
        
        return self.motion_frame
    
    def calc_full_frame(self):
        motion_frame = self.calc_motion_frame()
        if motion_frame is None:
            self.frame = None
            return None
        layered_frame = self.mf[0].layer_over_image(motion_frame, self.src_frame)
        
        self.frame = layered_frame
        return self.frame

# How much to scale outputs up by
upscale_factor = 1 # 6 to go from 360p to 2160p
upscale_method = cv2.INTER_NEAREST
# How much to scale down the display by
downscale_factor = 1 # 4 to go from 2160p to 720p
downscale_method = cv2.INTER_LINEAR


# Create the BadApple object
ba = BadApple(BadApple.Quality.SD60)

# Create the AppleMotionFlowMulti object
mfm = AppleMotionFlowMulti(
    ba,
    flow_layers=1,
    flow_iterations=3,
    flow_windows_count=6,
    flow_windows_min=7,
    flow_windows_max=35,
    flow_windows_balance=False
)

# Make output filenames
temp_filename = ba.name + "_temp" + ba.ext
new_filename = ba.name + "_edit" + ba.ext

# Delete existing outputs
try:
    os.remove(temp_filename)
except:
    pass
try:
    os.remove(new_filename)
except:
    pass

# Calculate video sizes
video_size = (round(ba.width*upscale_factor), round(ba.height*upscale_factor))
display_size = (round(ba.width/downscale_factor), round(ba.height/downscale_factor))

# Start writing new file
new_video = cv2.VideoWriter(
    filename=temp_filename,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=ba.fps,
    frameSize=video_size
)

# Make playback window
windowName = 'Bad Apple'
cv2.namedWindow(windowName)

# Get first frame of the bad apple video
frame1 = ba.read_frame()
mfm.set_next_src_frames(frame1)

# Play the video
user_stopped = False
while True:
    print("Processing frame {}/{}".format(ba.frame_num, ba.total_frames))
    # Get flow
    final_frame = mfm.calc_full_frame()
    
    # This means it could not read the frame 
    if final_frame is None:
         print("Could not read the frame, video is likely over.")   
         cv2.destroyWindow(windowName)
         ba.close()
         break
    
    # Display frame
    if downscale_factor != 1:
        display_frame = cv2.resize(final_frame, display_size, 0, 0, interpolation = downscale_method)
    else:
        display_frame = final_frame
    cv2.imshow(windowName, display_frame)
    
    # Scale frame for outputs
    if upscale_factor != 1:
        final_video_frame = cv2.resize(final_frame, video_size, 0, 0, interpolation = upscale_method)
    else:
        final_video_frame = final_frame
    
    # Save frame
    new_video.write(final_video_frame)
    
    # Exit hotkey
    stop_playing = False
    waitKey = (cv2.waitKey(1) & 0xFF)
    if waitKey == ord('q'): # If Q pressed
        stop_playing = True
    
    if stop_playing:
        print("Closing video and exiting...")
        user_stopped = True
        cv2.destroyWindow(windowName)
        ba.close()
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

