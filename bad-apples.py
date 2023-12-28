import cv2

from mflowm import MotionFlowMulti, CompositeMode, VideoReader
from bad_apple import BadApple, Quality


def run_any_video(
        filename,
        mode: CompositeMode,
        trails: bool = False,
        fade_speed: float = 2,
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


def run_bad_apple(
        mode: CompositeMode,
        quality: Quality = Quality.SD,
        trails: bool = False,
        fade_speed: float = 2,
        windows_balance: bool = False,
        output_scale=6,
        display_scale=1,
        scale_method=cv2.INTER_NEAREST
):
    # Create the BadApple object
    video_reader = BadApple(quality)

    # Create the MotionFlowMulti object
    mfm = MotionFlowMulti(
        video_reader,
        mode=mode,
        trails=trails,
        fade_speed=fade_speed,
        windows_balance=windows_balance
    )

    mfm.convert_to_file(
        output_scale=output_scale,
        display_scale=display_scale,
        output_scale_method=scale_method,
        display_scale_method=scale_method,
        window_title="Bad Apple"
    )


def main():
    #run(CompositeMode.NONE)
    #run(CompositeMode.SIMPLE)
    #run(CompositeMode.GLITCH)
    #run(CompositeMode.BROKEN_A)
    run_bad_apple(CompositeMode.BROKEN_B)
    '''
    run_any_video(
        filename="VID_20231227_053334025.mp4",
        mode=CompositeMode.GLITCH,
        pre_scale=0.125,
        fade_speed=0,
        display_scale=0.5,
        windows_balance=True,
        trails=True
    )
    '''


if __name__ == "__main__":
    main()
