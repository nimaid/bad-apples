import cv2
import numpy as np
import blend_modes as bm

from mflowm.files import VideoFile


def layer_motion_frames(top, bottom, clip: bool = False):
    if clip:
        final_frame = np.clip(np.maximum(top, bottom), 0, 256).astype(np.uint8)
    else:
        # Add alpha channel for blend_modes module
        bottom_alpha = cv2.cvtColor(bottom, cv2.COLOR_RGB2RGBA)
        top_alpha = cv2.cvtColor(top, cv2.COLOR_RGB2RGBA)

        # Convert to float for blend_modes module
        bottom_alpha_float = bottom_alpha / 255.0
        top_alpha_float = top_alpha / 255.0

        # Do the blending with lighten
        blend_frame = bm.lighten_only(top_alpha_float, bottom_alpha_float, 1.0)

        # Convert back to int
        final_frame_alpha = (blend_frame * 255).astype(np.uint8)

        # Strip alpha channel
        final_frame = cv2.cvtColor(final_frame_alpha, cv2.COLOR_RGBA2RGB)

    return final_frame

# Function used to layer motion colors over the source video
# Currently using difference
def layer_over_image(top, bottom):
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


class MotionFlowMulti:
    """A class to handle multi-pass motion flow computation"""
    def __init__(
            self,
            video_file: VideoFile,  # Existing VideoFile object
            num_windows: int = 7,  # Number of flow calculations to do at different sizes
            windows_min: int = 7,  # Relative to video scale, higher gets bigger motions but is blurrier
            windows_max: int = 35,  # Relative to video scale, higher gets bigger motions but is blurrier
            windows_balance: bool = True,  # If we want to divide each motion layer brightness based on number of windows
            layers: int = 1,  # Number of layers in computation, more is better but slower
            iterations: int = 3,  # Number of iterations per layer, more is better but slower
            poly_n: int = 7,
            poly_sigma: int = 1.5,
            blur_amount: int = 1.5,  # Relative to video scale and flow window
            fade_speed: int = 2  # Relative to FPS
    ):
        self.video_file = video_file
        self.num_windows = num_windows
        self.windows_balance = windows_balance
        self.layers = layers
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.blur_amount = blur_amount
        self.fade_speed = fade_speed

        self.window_sizes = [int(round(x)) for x in np.linspace(windows_min, windows_max, self.num_windows)]

        # Make image to fade with
        self.fade_amt = max(round(self.fade_speed / self.video_file.fps_scale), 1)
        self.img_sub = np.ones(self.video_file.shape) * self.fade_amt

        # Make HSV array
        self.hsv = np.zeros(self.video_file.shape).astype(np.uint8)
        self.hsv[..., 1] = 255  # Full saturation

        # Init frames
        self.frame = None
        self.src_frame = None
        self.prev_src_frame = None

        self.motion_frame = None
        self.prev_motion_frame = None

        # Init other vars
        self.flow = [None for x in self.window_sizes]

    def blur_px(self, window_size):
        blur_px = max(round(self.blur_amount * window_size), 1)
        if blur_px % 2 == 0:
            blur_px += 1

        return blur_px

    # Function to read next frame from video file
    def get_next_src_frame(self) -> bool:
        self.video_file.read_frame()
        # Check if frame could not be gotten
        if self.video_file.frame is None:
            return False
        self.prev_src_frame = self.src_frame
        self.src_frame = self.video_file.frame
        return True

    # Alternate way to set with external frames
    def set_next_src_frames(self, src_frame, prev_src_frame=None):
        if prev_src_frame is None:
            self.prev_src_frame = self.src_frame
        else:
            self.prev_src_frame = prev_src_frame
        self.src_frame = src_frame

    # Function to darken an image based on the fade amount
    def fade_img(self, img, make_new_fade=False):
        if make_new_fade:  # Don't use pre-computed array
            img_sub = np.ones_like(img) * self.fade_amt
        else:
            img_sub = self.img_sub
        return np.subtract(img, img_sub.astype(np.int16)).clip(0, 255).astype(np.uint8)

    def get_flow(
            self,
            first_frame,
            second_frame,
            window_size,
            prev_flow=None
    ):
        # Convert to gray
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)

        # Get flow
        if prev_flow is None:  # Oh god it SUCKS why doesn't it work never give it the flow for the love of god
            flow_in = None
            flow_opts = 0
        else:
            flow_in = prev_flow
            flow_opts = cv2.OPTFLOW_USE_INITIAL_FLOW

        flow = cv2.calcOpticalFlowFarneback(
            first_frame_gray,
            second_frame_gray,
            flow_in,
            pyr_scale=0.5,  # Layer "pyramid" size ratio
            levels=self.layers,
            winsize=window_size,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=flow_opts
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = ang * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_frame = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        # Smooth colors with a blur
        smooth_frame = cv2.GaussianBlur(flow_frame, (self.blur_px(window_size), self.blur_px(window_size)), 0)

        return smooth_frame

    # Computes the next motion flow frame
    def calc_motion_frame(self, old_frame=None, bad_fade=False, do_fade=True):
        result = self.get_next_src_frame()
        # If we haven't gotten the second frame yet, get it
        if self.prev_src_frame is None:
            result = not ((not result) | (not self.get_next_src_frame()))

        if not result:
            self.motion_frame = None
            return None

        # Compute the multi-pass optical flow motion frame
        motion_frame = np.zeros(self.video_file.shape).astype(np.uint8)
        for window_index, window_size in enumerate(self.window_sizes):
            motion_flow_frame = self.get_flow(
                first_frame=self.prev_src_frame,
                second_frame=self.src_frame,
                window_size=window_size
            )

            # Darken based on number of windows
            if self.windows_balance:
                motion_flow_frame = np.around(motion_flow_frame / self.num_windows).astype(np.uint8)
            # Layer over previous motion flow frames
            motion_frame = layer_motion_frames(motion_flow_frame, motion_frame)

        # Layer over old motion frame if it exists
        if old_frame is None or self.prev_motion_frame is None:  # We don't have both, just set the motion current frame be the first frame
            layered_motion_frame = motion_frame
        else:
            # Darken last motion frame
            if do_fade:
                if bad_fade:
                    fade_amt = 0.2
                    img_black = np.zeros_like(old_frame)
                    motion_frame_bg = cv2.addWeighted(old_frame, 1 - fade_amt, img_black, fade_amt, 0.0)
                else:
                    motion_frame_bg = self.fade_img(old_frame)
            else:
                motion_frame_bg = old_frame
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
        layered_frame = layer_over_image(self.src_frame, motion_frame)

        self.frame = layered_frame
        return self.frame
