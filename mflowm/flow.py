import os
import cv2
import logging
import numpy as np
from etatime import EtaBar
import ffmpeg

from mflowm.files import VideoReader
from mflowm.layer import LayerMode, layer_images


class CompositeMode:
    NONE = 0
    SIMPLE = 1
    GLITCH = 2
    BROKEN_A = 3
    BROKEN_B = 4


class MotionFlowMulti:
    """A class to handle multi-pass motion flow computation"""
    def __init__(
            self,
            video_file: VideoReader,  # Existing VideoFile object
            mode: CompositeMode = CompositeMode.NONE,  # How to combine the motion with the source
            trails: bool = False,
            num_windows: int = 7,  # Number of flow calculations to do at different sizes
            windows_min: int = 7,  # Relative to video scale, higher gets bigger motions but is blurrier
            windows_max: int = 35,  # Relative to video scale, higher gets bigger motions but is blurrier
            windows_balance: bool = False,  # If we want to divide each motion layer brightness based on number of windows
            layers: int = 1,  # Number of layers in computation, more is better but slower
            iterations: int = 3,  # Number of iterations per layer, more is better but slower
            poly_n: int = 7,
            poly_sigma: int = 1.5,
            blur_amount: int = 1.5,  # Relative to video scale and flow window
            fade_speed: float = 2  # Relative to FPS
    ):
        self.video_file = video_file
        self.mode = mode
        self.trails = trails
        self.num_windows = num_windows
        self.windows_balance = windows_balance
        self.layers = layers
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.blur_amount = blur_amount
        self.fade_speed = fade_speed

        if self.mode in [CompositeMode.BROKEN_A, CompositeMode.BROKEN_B] and not self.trails:
            logging.warning("Overriding trails setting to enable broken composite mode.")
            self.trails = True

        self.window_sizes = [int(round(x)) for x in np.linspace(windows_min, windows_max, self.num_windows)]

        # Make image to fade with
        self.fade_amt = round(self.fade_speed / self.video_file.fps_scale)

        # Make HSV array
        self.hsv = np.zeros(self.video_file.shape).astype(np.uint8)
        self.hsv[..., 1] = 255  # Full saturation

        # Init frames
        self.frame = None

        self.src_frame = None
        self.prev_src_frame = None

        self.motion_frame = None
        self.prev_motion_frame = None

    def _blur_px(self, window_size):
        blur_px = max(round(self.blur_amount * window_size), 1)
        if blur_px % 2 == 0:
            blur_px += 1

        return blur_px

    # Function to read next frame from video file
    def _get_next_src_frame(self) -> bool:
        self.video_file.read_frame()
        # Check if frame could not be gotten
        if self.video_file.frame is None:
            return False
        self.prev_src_frame = self.src_frame
        self.src_frame = self.video_file.frame
        return True

    # Function to darken an image based on the fade amount
    def _fade_img(self, img, bad_fade=False):
        if bad_fade:
            fade_amt = 0.2
            img_black = np.zeros_like(img)
            return cv2.addWeighted(img, 1 - fade_amt, img_black, fade_amt, 0.0)

        img_sub = np.ones_like(img) * self.fade_amt
        return np.subtract(img, img_sub.astype(np.int16)).clip(0, 255).astype(np.uint8)

    def _get_flow(
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
        if prev_flow is None:  # I don't even track the last flows b/c this looks bad when used
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
        self.hsv[..., 2] = cv2.normalize(src=mag, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        flow_frame = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        # Smooth colors with a blur
        smooth_frame = cv2.GaussianBlur(flow_frame, (self._blur_px(window_size), self._blur_px(window_size)), 0)

        return smooth_frame

    # Computes the next motion flow frame
    def _calc_motion_frame(self, old_frame=None, bad_fade=False, do_fade=True):
        result = self._get_next_src_frame()
        # If we haven't gotten the second frame yet, get it
        if self.prev_src_frame is None:
            result = result and self._get_next_src_frame()

        if not result:
            self.motion_frame = None
            return None

        # Compute the multi-pass optical flow motion frame
        motion_frame = np.zeros(self.video_file.shape).astype(np.uint8)
        for window_index, window_size in enumerate(self.window_sizes):
            motion_flow_frame = self._get_flow(
                first_frame=self.prev_src_frame,
                second_frame=self.src_frame,
                window_size=window_size
            )

            # Darken based on number of windows
            if self.windows_balance:
                motion_flow_frame = np.around(motion_flow_frame / self.num_windows).astype(np.uint8)
            # Layer over previous motion flow frames
            motion_frame = layer_images(motion_flow_frame, motion_frame, LayerMode.LIGHTEN)

        if self.motion_frame is not None:
            self.prev_motion_frame = self.motion_frame
        else:
            self.prev_motion_frame = None

        if old_frame is None:
            old_frame = self.prev_motion_frame

        # Layer over old motion frame if it exists
        if old_frame is None or not self.trails:  # Just set the motion current frame be the first frame
            layered_motion_frame = motion_frame
        else:
            # Darken last motion frame
            if do_fade:
                motion_frame_bg = self._fade_img(old_frame, bad_fade=bad_fade)
            else:
                motion_frame_bg = old_frame
            # Add over last motion frame by blending with lighten
            layered_motion_frame = layer_images(motion_frame, motion_frame_bg, LayerMode.CLIP)



        self.motion_frame = layered_motion_frame
        return self.motion_frame

    def get_next_frame(self):
        match self.mode:
            case CompositeMode.BROKEN_A:
                motion_frame = self._calc_motion_frame(
                    old_frame=self.frame,
                    do_fade=False
                )
            case CompositeMode.BROKEN_B:
                motion_frame = self._calc_motion_frame(
                    old_frame=self.frame,
                    bad_fade=True
                )
            case _:
                motion_frame = self._calc_motion_frame()

        if motion_frame is None:
            self.frame = None
            return None

        # Layer the motion over the source
        match self.mode:
            case CompositeMode.GLITCH:
                layered_frame = layer_images(self.src_frame, motion_frame, LayerMode.DIFFERENCE)
                final_frame = layer_images(layered_frame, motion_frame, LayerMode.INVERT_CLIP)
            case CompositeMode.SIMPLE:
                final_frame = layer_images(self.src_frame, motion_frame, LayerMode.DIFFERENCE)
            case CompositeMode.BROKEN_A | CompositeMode.BROKEN_B:
                final_frame = layer_images(self.src_frame, motion_frame, LayerMode.INVERT_CLIP)
            case _:
                final_frame = motion_frame  # No layering (only motion)

        self.frame = final_frame
        return self.frame

    def convert_to_file(
            self,
            filename_suffix="_flow",
            output_scale=1,
            display_scale=1,
            output_scale_method=cv2.INTER_NEAREST,
            display_scale_method=cv2.INTER_NEAREST,
            window_title="Python MotionFlowMulti (mflowm)",
            quit_key="q",
            fade_to_black=True
    ):
        # Make output filenames
        filename = self.video_file.name + filename_suffix + ".mp4"
        temp_filename = self.video_file.name + "_temp" + ".mp4"

        # Delete existing outputs
        try:
            os.remove(temp_filename)
        except FileNotFoundError:
            pass

        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

        # Calculate video sizes
        video_size = (round(self.video_file.width * output_scale), round(self.video_file.height * output_scale))
        display_size = (round(self.video_file.width * display_scale), round(self.video_file.height * display_scale))

        # Start writing new file
        new_video = cv2.VideoWriter(
            filename=temp_filename,
            fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
            fps=self.video_file.fps,
            frameSize=video_size
        )

        # Make playback window
        window_name = f"{window_title} (press {quit_key.upper()} to quit)"
        cv2.namedWindow(window_name)

        # Play the video
        user_stopped = False
        final_video_frame = None
        for i in EtaBar(range(self.video_file.total_frames), bar_format="{l_bar}{bar}{r_barL}"):
            try:
                final_frame = self.get_next_frame()
                # This means it could not read the frame (should never happen)
                if final_frame is None:
                    logging.warning("Could not read the frame, video is likely over.")
                    cv2.destroyWindow(window_name)
                    self.video_file.close()
                    new_video.release()
                    return

                # Display frame
                if display_scale != 1:
                    display_frame = cv2.resize(final_frame, display_size, 0, 0, interpolation=output_scale_method)
                else:
                    display_frame = final_frame
                cv2.imshow(window_name, display_frame)

                # Scale frame for outputs
                if output_scale != 1:
                    final_video_frame = cv2.resize(final_frame, video_size, 0, 0, interpolation=display_scale_method)
                else:
                    final_video_frame = final_frame

                # Save frame
                new_video.write(final_video_frame)

                # Exit hotkey
                stop_playing = False
                wait_key = (cv2.waitKey(1) & 0xFF)
                if wait_key == ord(quit_key):  # If quit key pressed
                    logging.warning(f"User interrupted rendering process. ({quit_key.upper()})")
                    stop_playing = True
            except KeyboardInterrupt:
                logging.warning("User interrupted rendering process. (CTRL + C)")
                stop_playing = True

            if stop_playing:
                logging.warning("Closing video and exiting...")
                user_stopped = True
                cv2.destroyWindow(window_name)
                self.video_file.close()
                break

        # Add fade frames and keep fading until it is fully black
        if fade_to_black:
            fade_frames = 0
            if not user_stopped:
                logging.info("Adding extra frames so that the video fades fully to black...")
                while np.sum(final_video_frame, axis=None) != 0:  # While the last frame isn't completely black
                    try:
                        logging.info(f"Fade frame {fade_frames + 1}")
                        final_video_frame = self._fade_img(final_video_frame)  # Fade the image
                        new_video.write(final_video_frame)  # Write the image
                        fade_frames += 1
                    except KeyboardInterrupt:
                        logging.warning("User interrupted fading process. (CTRL + C)")
                        user_stopped = True
                        break
                if not user_stopped:
                    logging.info("Video is now fully black!")

        # Save new video
        new_video.release()

        # If user quit, stop here
        if user_stopped:
            return

        # Mux original audio and new video together (lossless, copy streams)
        logging.info("Adding audio...")
        audio_original = ffmpeg.input(self.video_file.filename).audio
        video_new = ffmpeg.input(temp_filename).video
        video_muxed = ffmpeg.output(audio_original, video_new, filename, vcodec='copy', acodec='copy')
        ffmpeg_result = video_muxed.run()
        if os.path.exists(filename):
            os.remove(temp_filename)  # Delete the temp file
        logging.info("Done converting video!")
