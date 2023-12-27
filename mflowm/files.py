import os
import cv2


class VideoFile:
    def __init__(
            self,
            filename: str
    ):
        self.filename = filename
        self.filename_full = os.path.realpath(self.filename)

        # Open the video file
        self.video = None
        self.open()

        # Get properties from video
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.size = (self.width, self.height)
        self.shape = (self.height, self.width, 3)

        # Calculate the scales for the motion flow object
        self.img_scale = self.height / 360
        self.fps_scale = self.fps / 30

        # Initialize attributes
        self.frame = None
        self.frame_num = 0

    def __del__(self):
        self.close()

    # Function to open the source video
    def open(self) -> None:
        # Check if already opened
        if self.video is not None:
            return

        # Make capture object for playback
        video = cv2.VideoCapture(self.filename_full)

        # Check that the capture object is ready
        if not video.isOpened():
            raise Exception("Could not open video file")
        else:
            self.video = video

    # Function to close the source video
    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    # Function to read next frame
    def read_frame(self) -> None:
        ret, frame = self.video.read()
        if ret:
            self.frame_num += 1
            self.frame = frame
        else:  # Cant read frame, video is probably over
            self.frame_num = -1
            self.frame = None
