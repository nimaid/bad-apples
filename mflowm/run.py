import cv2

from mflowm import MotionFlowMulti, CompositeMode, VideoReader


def run(
        filename,
        mode: CompositeMode,
        trails: bool = False,
        fade_speed: float | None = 2,
        windows_balance: bool = False,
        pre_scale: float = 1,
        display_scale: float = 1,
        scale_method=cv2.INTER_NEAREST
):
    # Create the BadApple object
    video_reader = VideoReader(filename, scale=pre_scale)

    # Create the MotionFlowMulti object
    mfm = MotionFlowMulti(
        video_reader,
        mode=mode,
        trails=trails,
        fade_speed=fade_speed,
        windows_balance=windows_balance,
    )

    mfm.convert_to_file(
        output_scale=(1 / pre_scale),
        display_scale=(1 / pre_scale) * display_scale,
        output_scale_method=scale_method,
        display_scale_method=scale_method
    )
