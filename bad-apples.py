from mflowm import MotionFlowMulti, CompositeMode
from bad_apple import BadApple, Quality


def run(
        mode: CompositeMode,
        quality=Quality.SD,
        output_scale=6,
        display_scale=1
):
    # Create the BadApple object
    video_reader = BadApple(quality)

    # Create the MotionFlowMulti object
    mfm = MotionFlowMulti(
        video_reader,
        mode=mode
    )

    mfm.convert_to_file(
        output_scale=output_scale,
        display_scale=display_scale,
        window_title="Bad Apple"
    )


def main():
    #run(CompositeMode.NONE)
    #run(CompositeMode.SIMPLE)
    #run(CompositeMode.GLITCH)
    #run(CompositeMode.BROKEN_A)
    run(CompositeMode.BROKEN_B)


if __name__ == "__main__":
    main()
