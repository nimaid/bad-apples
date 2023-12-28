import os
import sys
from enum import Enum
import cv2
import numpy as np
import ffmpeg
from etatime import EtaBar

from mflowm import BadApple, MotionFlowMulti, CompositeMode


def run(
        mode: CompositeMode,
        quality=BadApple.Quality.SD,
        trails=None,
        upscale_factor=6,
        downscale_factor=1,
        upscale_method=cv2.INTER_NEAREST,
        downscale_method=cv2.INTER_CUBIC,
        fade_speed=2
):
    # Create the BadApple object
    video_reader = BadApple(quality)

    # Create the MotionFlowMulti object
    if trails is not None:
        mfm = MotionFlowMulti(
            video_reader,
            mode=mode,
            windows_balance=False,
            trails=trails,
            fade_speed=fade_speed
        )
    else:
        mfm = MotionFlowMulti(
            video_reader,
            mode=mode,
            windows_balance=False,
            fade_speed=fade_speed
        )

    # Make output filenames
    temp_filename = video_reader.name + "_temp" + video_reader.ext
    new_filename = video_reader.name + "_edit" + video_reader.ext

    # Delete existing outputs
    try:
        os.remove(temp_filename)
    except FileNotFoundError:
        pass

    try:
        os.remove(new_filename)
    except FileNotFoundError:
        pass

    # Calculate video sizes
    video_size = (round(video_reader.width * upscale_factor), round(video_reader.height * upscale_factor))
    display_size = (round(video_reader.width / downscale_factor), round(video_reader.height / downscale_factor))

    # Start writing new file
    if video_reader.ext == ".mp4":
        new_video = cv2.VideoWriter(
            filename=temp_filename,
            fourcc=cv2.VideoWriter.fourcc("m", "p", "4", "v"),
            fps=video_reader.fps,
            frameSize=video_size
        )
    else:
        raise Exception(f"File extension not supported: {video_reader.ext}")

    # Make playback window
    quit_key = "q"
    window_name = f"Bad Apple (press {quit_key.upper()} to quit)"
    cv2.namedWindow(window_name)

    # Play the video
    user_stopped = False
    final_video_frame = None
    previous_frame = None
    for i in EtaBar(range(video_reader.total_frames), bar_format="{l_bar}{bar}{r_barL}", file=sys.stdout):
        try:
            final_frame = mfm.calc_full_frame()
            # This means it could not read the frame (should never happen)
            if final_frame is None:
                print("\nCould not read the frame, video is likely over.")
                cv2.destroyWindow(window_name)
                video_reader.close()
                break

            # Display frame
            if downscale_factor != 1:
                display_frame = cv2.resize(final_frame, display_size, 0, 0, interpolation=downscale_method)
            else:
                display_frame = final_frame
            cv2.imshow(window_name, display_frame)

            # Scale frame for outputs
            if upscale_factor != 1:
                final_video_frame = cv2.resize(final_frame, video_size, 0, 0, interpolation=upscale_method)
            else:
                final_video_frame = final_frame

            # Save frame
            new_video.write(final_video_frame)

            # Exit hotkey
            stop_playing = False
            wait_key = (cv2.waitKey(1) & 0xFF)
            if wait_key == ord(quit_key):  # If quit key pressed
                print("\nUser interrupted rendering process. ({})".format(quit_key.upper()))
                stop_playing = True
        except KeyboardInterrupt:
            print("\nUser interrupted rendering process. (CTRL + C)")
            stop_playing = True

        if stop_playing:
            print("\nClosing video and exiting...")
            user_stopped = True
            cv2.destroyWindow(window_name)
            video_reader.close()
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
    audio_original = ffmpeg.input(video_reader.filename).audio
    video_new = ffmpeg.input(temp_filename).video_file
    video_muxed = ffmpeg.output(audio_original, video_new, new_filename, vcodec='copy', acodec='copy')
    ffmpeg_result = video_muxed.run()
    if os.path.exists(new_filename):
        os.remove(temp_filename)  # Delete the temp file
    print("\nAdded audio!")


def main():
    #run(CompositeMode.NONE)
    #run(CompositeMode.SIMPLE)
    #run(CompositeMode.GLITCH)
    #run(CompositeMode.BROKEN_A)
    run(CompositeMode.BROKEN_B)


if __name__ == "__main__":
    main()
