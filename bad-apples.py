import cv2

import mflowm.run
from mflowm import MotionFlowMulti, CompositeMode, VideoReader
from bad_apple import BadApple, Quality

def run_bad_apple(
        mode: CompositeMode,
        quality: Quality = Quality.SD,
        trails: bool = False,
        fade_speed: float | None = 2,
        windows_balance: bool = False,
        output_scale: float = 6,
        display_scale: float = 1,
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
    #run_bad_apple(CompositeMode.NONE)
    #run_bad_apple(CompositeMode.SIMPLE)
    #run_bad_apple(CompositeMode.GLITCH)
    #run_bad_apple(CompositeMode.BROKEN_A)
    run_bad_apple(CompositeMode.BROKEN_B)
    '''
    run_bad_apple(
        mode=CompositeMode.GLITCH,
        fade_speed=None,
        windows_balance=True,
        trails=True,
        scale_method=cv2.INTER_NEAREST
    )
    '''
    '''
    mflowm.run(
        filename="VID_20231228_140141022.mp4",
        mode=CompositeMode.SIMPLE,
        pre_scale=1,
        fade_speed=None,
        display_scale=0.5,
        windows_balance=False,
        trails=True,
        scale_method=cv2.INTER_NEAREST
    )
    '''


if __name__ == "__main__":
    main()
