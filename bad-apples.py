import os
import sys
import urllib.request
from enum import Enum
import hashlib
import cv2
import numpy as np
import blend_modes as bm
import ffmpeg
from etatime import EtaBar

from mflowm import MotionFlowMulti, layer_over_image

# !!!! WARNING: VISUAL HAZARD AHEAD!!! BAD CODE!!! AVERT YOUR EYES!!!


# A class to represent the Bad Apple video
# Capable of multiple video qualities
# Will download and verify missing files
class BadApple:
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

    def __init__(
            self,
            quality=None
    ):
        if quality is None:  # default to BadApple.Quality.SD
            self.quality = self.Quality.SD
        else:
            self.quality = quality

        self.ext = ".mp4"
        if self.quality == self.Quality.SD:
            self.name = "bad_apple"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple.mp4"
            self.sha1 = "d248203e4f8a88433bee75cf9d0e746386ba4b1b"
            self.img_scale = 1  # 360p
            self.fps_scale = 1  # 30 FPS
        elif self.quality == self.Quality.SD60:
            self.name = "bad_apple@60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%4060fps.mp4"
            self.sha1 = "f154318c4049b665aa4fa4dc819b10c2c34ff97e"
            self.img_scale = 1  # 360p
            self.fps_scale = 2  # 60 FPS
        elif self.quality == self.Quality.HD:
            self.name = "bad_apple@720p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p.mp4"
            self.sha1 = "333bae3a21b4e514e06f5a6b1104dfb0c698411e"
            self.img_scale = 2  # 720p
            self.fps_scale = 1  # 30 FPS
        elif self.quality == self.Quality.HD60:
            self.name = "bad_apple@720p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%40720p60fps.mp4"
            self.sha1 = "15c22498e6abf3fb0f7ca73d39d281a3e5c0c706"
            self.img_scale = 2  # 720p
            self.fps_scale = 2  # 60 FPS
        elif self.quality == self.Quality.FHD:
            self.name = "bad_apple@1080p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p.mp4"
            self.sha1 = "b8fef140406312d4bc2a51936d7de9c47fe02e8b"
            self.img_scale = 3  # 1080p
            self.fps_scale = 1  # 30 FPS
        elif self.quality == self.Quality.FHD60:
            self.name = "bad_apple@1080p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401080p60fps.mp4"
            self.sha1 = "549491981229b937dc5f3987851d343a456828f2"
            self.img_scale = 3  # 1080p
            self.fps_scale = 2  # 60 FPS
        elif self.quality == self.Quality.QHD:
            self.name = "bad_apple@1440p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p.mp4"
            self.sha1 = "012425b863987ef84e4aafabbb66998dd6e15d51"
            self.img_scale = 4  # 1440p
            self.fps_scale = 1  # 30 FPS
        elif self.quality == self.Quality.QHD60:
            self.name = "bad_apple@1440p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%401440p60fps.mp4"
            self.sha1 = "6204b3173ec745f4c583b6dde11f858a7886b8d0"
            self.img_scale = 4  # 1440p
            self.fps_scale = 2  # 60 FPS
        elif self.quality == self.Quality.UHD:
            self.name = "bad_apple@2160p"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p.mp4"
            self.sha1 = "028ec64b3c909a92b6532b32a2473f735667feb0"
            self.img_scale = 6  # 2160p
            self.fps_scale = 1  # 30 FPS
        elif self.quality == self.Quality.UHD60:
            self.name = "bad_apple@2160p60fps"
            self.url = "https://archive.org/download/bad-apple-resources/bad_apple%402160p60fps.mp4"
            self.sha1 = "d5dcaef680abbff71c0e9fb9de130d45a4ba2cb7"
            self.img_scale = 6  # 2160p
            self.fps_scale = 2  # 60 FPS
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
        else:  # Cant read frame, video is probably over
            self.frame_num = -1
            self.frame = None
            return None


# A class to act as a stand-in for a BadApple object, but scaled down
class BadAppleResizeDummy:
    def __init__(
            self,
            ba,
            shrink_ratio=3  # 3 to go from 2160p to 720p
    ):
        self.ba = ba
        self.shrink_ratio = shrink_ratio

        self.quality = self.ba.quality

        self.ext = self.ba.ext

        self.name = self.ba.name
        self.url = self.ba.url
        self.sha1 = self.ba.sha1
        self.img_scale = self.ba.img_scale / self.shrink_ratio
        self.fps_scale = self.ba.fps_scale

        self.filename = self.ba.filename
        self.width = round(self.ba.width / self.shrink_ratio)
        self.height = round(self.ba.height / self.shrink_ratio)
        self.size = (self.width, self.height)

        self.fps = self.ba.fps
        self.shape = (self.height, self.width, 3)

        self.filename_full = self.ba.filename_full

        self.video = self.ba.video_file

        self.total_frames = self.ba.total_frames

        self.frame = self.ba.frame
        self.frame_num = self.ba.frame_num

    # Function to open the source video
    def open(self):
        result = self.ba.open()
        self.video = self.ba.video_file
        return result

    # Function to close the source video
    def close(self):
        self.ba.close()

    # Function to read next frame in parent and resize
    def read_frame(self):
        frame = self.ba.read_frame()
        self.frame_num = self.ba.frame_num
        if frame is None:
            self.frame = None
            return None
        else:
            self.frame = cv2.resize(frame, self.size, 0, 0, interpolation=cv2.INTER_LINEAR)
            return self.frame


class LayerMode(Enum):
    GLITCH = 0
    SIMPLE = 1
    BROKEN = 2
    VERY_BROKEN = 3
    NONE = 4


def run(
        layer_mode,
        quality=BadApple.Quality.SD,
        upscale_factor=6,
        downscale_factor=1,
        upscale_method=cv2.INTER_NEAREST,
        downscale_method=cv2.INTER_CUBIC,
        fade_speed=30
):
    # Create the 720p BadApple object
    ba = BadApple(quality)

    # Create the AppleMotionFlowMulti object
    mfm = MotionFlowMulti(
        ba,
        windows_balance=False,
        fade_speed=fade_speed
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
    video_size = (round(ba.width * upscale_factor), round(ba.height * upscale_factor))
    display_size = (round(ba.width / downscale_factor), round(ba.height / downscale_factor))

    # Start writing new file
    new_video = cv2.VideoWriter(
        filename=temp_filename,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=ba.fps,
        frameSize=video_size
    )

    # Make playback window
    quit_key = "q"
    windowName = "Bad Apple (press {} to quit)".format(quit_key.upper())
    cv2.namedWindow(windowName)

    # Get first frame of the bad apple video
    frame1 = ba.read_frame()
    mfm.set_next_src_frames(frame1)

    # Play the video
    user_stopped = False
    final_video_frame = None
    previous_frame = None
    for i in EtaBar(range(ba.total_frames), bar_format="{l_bar}{bar}{r_barL}", file=sys.stdout):
        try:
            # Get motion frame
            if layer_mode == LayerMode.BROKEN:
                motion_frame = mfm.calc_motion_frame(old_frame=previous_frame, bad_fade=False, do_fade=False)
            elif layer_mode == LayerMode.VERY_BROKEN:
                motion_frame = mfm.calc_motion_frame(old_frame=previous_frame, bad_fade=True)
            else:
                motion_frame = mfm.calc_motion_frame()

            # This means it could not read the frame
            if motion_frame is None:
                print("\nCould not read the frame, video is likely over.")
                cv2.destroyWindow(windowName)
                ba.close()
                break

            # Layer the motion over the source
            match layer_mode:
                case LayerMode.GLITCH:
                    layered_frame = layer_over_image(mfm.video_file.frame, motion_frame)
                    # Glitchy clipping effect
                    final_frame = np.clip(
                        1 - np.multiply(1 - layered_frame, 1 - motion_frame),
                        0,
                        256).astype(np.uint8)
                case LayerMode.SIMPLE:
                    # Layered over source
                    final_frame = layer_over_image(mfm.video_file.frame, motion_frame)
                case LayerMode.BROKEN:
                    final_frame = np.clip(
                        1 - np.multiply(1 - motion_frame, 1 - mfm.video_file.frame),
                        0,
                        256).astype(np.uint8)

                    previous_frame = final_frame
                case LayerMode.VERY_BROKEN:
                    final_frame = np.clip(
                        1 - np.multiply(1 - motion_frame, 1 - mfm.video_file.frame),
                        0,
                        256).astype(np.uint8)

                    previous_frame = final_frame
                case _:
                    final_frame = motion_frame  # No layering (only motion)

            # Display frame
            if downscale_factor != 1:
                display_frame = cv2.resize(final_frame, display_size, 0, 0, interpolation=downscale_method)
            else:
                display_frame = final_frame
            cv2.imshow(windowName, display_frame)

            # Scale frame for outputs
            if upscale_factor != 1:
                final_video_frame = cv2.resize(final_frame, video_size, 0, 0, interpolation=upscale_method)
            else:
                final_video_frame = final_frame

            # Save frame
            new_video.write(final_video_frame)

            # Exit hotkey
            stop_playing = False
            waitKey = (cv2.waitKey(1) & 0xFF)
            if waitKey == ord(quit_key):  # If quit key pressed
                print("\nUser interrupted rendering process. ({})".format(quit_key.upper()))
                stop_playing = True
        except KeyboardInterrupt:
            print("\nUser interrupted rendering process. (CTRL + C)")
            stop_playing = True

        if stop_playing:
            print("\nClosing video and exiting...")
            user_stopped = True
            cv2.destroyWindow(windowName)
            ba.close()
            break

    # Add fade frames and keep fading until it is fully black
    fade_frames = 0
    if not user_stopped:
        print("\nAdding extra frames so that the video fades fully to black...")
        while np.sum(final_video_frame, axis=None) != 0:  # While the last frame isn't completely black
            try:
                print("Fade frame {}".format(fade_frames + 1))
                final_video_frame = mfm.fade_img(final_video_frame, make_new_fade=True)  # Fade the image
                new_video.write(final_video_frame)  # Write the image
                fade_frames += 1
            except KeyboardInterrupt:
                print("User interrupted fading process. (CTRL + C)")
                user_stopped = True
                break
        if not user_stopped:
            print("Video is now fully black!")

    # Save new video
    new_video.release()

    # If user quit, stop here
    if user_stopped:
        exit()

    # Mux original audio and new video together (lossless, copy streams)
    print("\nAdding audio...\n")
    audio_original = ffmpeg.input(ba.filename).audio
    video_new = ffmpeg.input(temp_filename).video_file
    video_muxed = ffmpeg.output(audio_original, video_new, new_filename, vcodec='copy', acodec='copy')
    ffmpeg_result = video_muxed.run()
    if os.path.exists(new_filename):
        os.remove(temp_filename)  # Delete the temp file
    print("\nAdded audio!")


def main():
    #run(LayerMode.GLITCH)
    run(LayerMode.SIMPLE)
    #run(LayerMode.BROKEN, quality=BadApple.Quality.FHD60)
    #run(LayerMode.VERY_BROKEN)
    #run(LayerMode.NONE)


if __name__ == "__main__":
    main()
