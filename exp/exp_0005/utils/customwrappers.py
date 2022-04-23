from gym import error, logger
import numpy as np
from io import StringIO
import tempfile
import subprocess
import pkgutil
import os.path
import json
import distutils.version
import distutils.spawn
import os
from typing import Callable, Optional

import gym
from gym import logger
# from gym.wrappers.monitoring import video_recorder


def capped_cubic_video_schedule(episode_id):  # 条件
    # if episode_id < 10:
    #     return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    # else:
    return episode_id % 10 == 0


class CustomRecordVideo(gym.Wrapper):
    def __init__(
        self,
        env,
        video_folder: str,
        init_episode=0,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "mario_rl",
    ):
        super().__init__(env)
        # 条件設定
        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule
        trigger_count = sum(x is not None for x in [
                            episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = None
        # video_folderを作成
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = init_episode
        self.frames = []

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        if not self.recording and self._video_enabled():
            self.start_video_recorder()
            self.frames = []
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()
        # video nameを付ける
        # video_name = f"{self.name_prefix}-step-{self.step_id}"
        # if self.episode_trigger:
        #     video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        # base_path = os.path.join(self.video_folder, video_name)
        # VideoRecoderに投げる
        # self.video_recorder = VideoRecorder(
        #     env=self.env,
        #     base_path=base_path,
        #     metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        # )
        # capture_frameてやつを行う。
        # self.video_recorder.capture_frame()
        # img = self.env.render()
        # self.frames.append(img)
        # recorded_framesでvideoの長さを管理？
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    # actionが入力されたときの反応
    def step(self, action):
        observations, rewards, dones, infos = super().step(action)

        video = None
        if infos['flag_get']:
            dones = True

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            # capture_frameてやつを行う。
            self.frames.append(observations)
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    video = self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        video = self.close_video_recorder()
                elif dones[0]:
                    video = self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()
        infos['video'] = video
        return observations, rewards, dones, infos

    def close_video_recorder(self) -> None:
        self.recording = False
        self.recorded_frames = 1
        if len(self.frames):
            video = np.stack(self.frames).transpose(0, 3, 1, 2)
            return video
        else:
            return None

    def close(self):
        super().close()
        self.close_video_recorder()

    def __del__(self):
        self.close_video_recorder()


# def touch(path):
#     open(path, "a").close()

# class VideoRecorder:
#     """VideoRecorder renders a nice movie of a rollout, frame by frame. It
#     comes with an `enabled` option so you can still use the same code
#     on episodes where you don't want to record video.
#     Note:
#         You are responsible for calling `close` on a created
#         VideoRecorder, or else you may leak an encoder process.
#     Args:
#         env (Env): Environment to take video of.
#         path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
#         base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
#         metadata (Optional[dict]): Contents to save to the metadata file.
#         enabled (bool): Whether to actually record video, or just no-op (for convenience)
#     """

#     def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None):
#         modes = env.metadata.get("render_modes", [])

#         # backward-compatibility mode:
#         backward_compatible_mode = env.metadata.get("render.modes", [])
#         if len(modes) == 0 and len(backward_compatible_mode) > 0:
#             logger.deprecation(
#                 '`env.metadata["render.modes"] is marked as deprecated and will be replaced with `env.metadata["render_modes"]` '
#                 "see https://github.com/openai/gym/pull/2654 for more details"
#             )
#             modes = backward_compatible_mode

#         self._async = env.metadata.get("semantics.async")
#         self.enabled = enabled
#         self._closed = False

#         # Don't bother setting anything else if not enabled
#         if not self.enabled:
#             return

#         self.ansi_mode = False
#         if "rgb_array" not in modes:
#             if "ansi" in modes:
#                 self.ansi_mode = True
#             else:
#                 logger.info(
#                     f'Disabling video recorder because {env} neither supports video mode "rgb_array" nor "ansi".'
#                 )
#                 # Whoops, turns out we shouldn't be enabled after all
#                 self.enabled = False
#                 return

#         if path is not None and base_path is not None:
#             raise error.Error(
#                 "You can pass at most one of `path` or `base_path`.")

#         self.last_frame = None
#         self.env = env

#         required_ext = ".json" if self.ansi_mode else ".mp4"
#         if path is None:
#             if base_path is not None:
#                 # Base path given, append ext
#                 path = base_path + required_ext
#             else:
#                 # Otherwise, just generate a unique filename
#                 with tempfile.NamedTemporaryFile(
#                     suffix=required_ext, delete=False
#                 ) as f:
#                     path = f.name
#         self.path = path

#         path_base, actual_ext = os.path.splitext(self.path)

#         if actual_ext != required_ext:
#             hint = (
#                 " HINT: The environment is text-only, therefore we're recording its text output in a structured JSON format."
#                 if self.ansi_mode
#                 else ""
#             )
#             raise error.Error(
#                 f"Invalid path given: {self.path} -- must have file extension {required_ext}.{hint}"
#             )
#         # Touch the file in any case, so we know it's present. (This
#         # corrects for platform platform differences. Using ffmpeg on
#         # OS X, the file is precreated, but not on Linux.
#         touch(path)

#         self.frames_per_sec = env.metadata.get("render_fps", 30)
#         self.output_frames_per_sec = env.metadata.get(
#             "render_fps", self.frames_per_sec)

#         # backward-compatibility mode:
#         self.backward_compatible_frames_per_sec = env.metadata.get(
#             "video.frames_per_second", self.frames_per_sec
#         )
#         self.backward_compatible_output_frames_per_sec = env.metadata.get(
#             "video.output_frames_per_second", self.output_frames_per_sec
#         )
#         if self.frames_per_sec != self.backward_compatible_frames_per_sec:
#             logger.deprecation(
#                 '`env.metadata["video.frames_per_second"] is marked as deprecated and will be replaced with `env.metadata["render_fps"]` '
#                 "see https://github.com/openai/gym/pull/2654 for more details"
#             )
#             self.frames_per_sec = self.backward_compatible_frames_per_sec
#         if self.output_frames_per_sec != self.backward_compatible_output_frames_per_sec:
#             logger.deprecation(
#                 '`env.metadata["video.output_frames_per_second"] is marked as deprecated and will be replaced with `env.metadata["render_fps"]` '
#                 "see https://github.com/openai/gym/pull/2654 for more details"
#             )
#             self.output_frames_per_sec = self.backward_compatible_output_frames_per_sec

#         self.encoder = None  # lazily start the process
#         self.broken = False

#         # Dump metadata
#         self.metadata = metadata or {}
#         self.metadata["content_type"] = (
#             "video/vnd.openai.ansivid" if self.ansi_mode else "video/mp4"
#         )
#         self.metadata_path = f"{path_base}.meta.json"
#         self.write_metadata()

#         logger.info("Starting new video recorder writing to %s", self.path)
#         self.empty = True

#     @property
#     def functional(self):
#         return self.enabled and not self.broken

#     def capture_frame(self):
#         """Render the given `env` and add the resulting frame to the video."""
#         if not self.functional:
#             return
#         if self._closed:
#             logger.warn(
#                 "The video recorder has been closed and no frames will be captured anymore."
#             )
#             return
#         logger.debug("Capturing video frame: path=%s", self.path)

#         render_mode = "ansi" if self.ansi_mode else "rgb_array"
#         frame = self.env.render(mode=render_mode)

#         if frame is None:
#             if self._async:
#                 return
#             else:
#                 # Indicates a bug in the environment: don't want to raise
#                 # an error here.
#                 logger.warn(
#                     "Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s",
#                     self.path,
#                     self.metadata_path,
#                 )
#                 self.broken = True
#         else:
#             self.last_frame = frame
#             if self.ansi_mode:
#                 self._encode_ansi_frame(frame)
#             else:
#                 self._encode_image_frame(frame)

#     def close(self):
#         """Flush all data to disk and close any open frame encoders."""
#         if not self.enabled or self._closed:
#             return

#         if self.encoder:
#             logger.debug("Closing video encoder: path=%s", self.path)
#             self.encoder.close()
#             self.encoder = None
#         else:
#             # No frames captured. Set metadata, and remove the empty output file.
#             os.remove(self.path)

#             if self.metadata is None:
#                 self.metadata = {}
#             self.metadata["empty"] = True

#         # If broken, get rid of the output file, otherwise we'd leak it.
#         if self.broken:
#             logger.info(
#                 "Cleaning up paths for broken video recorder: path=%s metadata_path=%s",
#                 self.path,
#                 self.metadata_path,
#             )

#             # Might have crashed before even starting the output file, don't try to remove in that case.
#             if os.path.exists(self.path):
#                 os.remove(self.path)

#             if self.metadata is None:
#                 self.metadata = {}
#             self.metadata["broken"] = True

#         self.write_metadata()

#         # Stop tracking this for autoclose
#         self._closed = True

#     def write_metadata(self):
#         with open(self.metadata_path, "w") as f:
#             json.dump(self.metadata, f)

#     def __del__(self):
#         # Make sure we've closed up shop when garbage collecting
#         self.close()

#     def _encode_ansi_frame(self, frame):
#         if not self.encoder:
#             self.encoder = TextEncoder(self.path, self.frames_per_sec)
#             self.metadata["encoder_version"] = self.encoder.version_info
#         self.encoder.capture_frame(frame)
#         self.empty = False

#     def _encode_image_frame(self, frame):
#         if not self.encoder:
#             self.encoder = ImageEncoder(
#                 self.path, frame.shape, self.frames_per_sec, self.output_frames_per_sec
#             )
#             self.metadata["encoder_version"] = self.encoder.version_info

#         try:
#             self.encoder.capture_frame(frame)
#         except error.InvalidFrame as e:
#             logger.warn(
#                 "Tried to pass invalid video frame, marking as broken: %s", e)
#             self.broken = True
#         else:
#             self.empty = False


# class TextEncoder:
#     """Store a moving picture made out of ANSI frames. Format adapted from
#     https://github.com/asciinema/asciinema/blob/master/doc/asciicast-v1.md"""

#     def __init__(self, output_path, frames_per_sec):
#         self.output_path = output_path
#         self.frames_per_sec = frames_per_sec
#         self.frames = []

#     def capture_frame(self, frame):
#         string = None
#         if isinstance(frame, str):
#             string = frame
#         elif isinstance(frame, StringIO):
#             string = frame.getvalue()
#         else:
#             raise error.InvalidFrame(
#                 f"Wrong type {type(frame)} for {frame}: text frame must be a string or StringIO"
#             )

#         frame_bytes = string.encode("utf-8")

#         if frame_bytes[-1:] != b"\n":
#             raise error.InvalidFrame(
#                 f'Frame must end with a newline: """{string}"""')

#         if b"\r" in frame_bytes:
#             raise error.InvalidFrame(
#                 f'Frame contains carriage returns (only newlines are allowed: """{string}"""'
#             )

#         self.frames.append(frame_bytes)

#     def close(self):
#         # frame_duration = float(1) / self.frames_per_sec
#         frame_duration = 0.5

#         # Turn frames into events: clear screen beforehand
#         # https://rosettacode.org/wiki/Terminal_control/Clear_the_screen#Python
#         # https://rosettacode.org/wiki/Terminal_control/Cursor_positioning#Python
#         clear_code = b"%c[2J\033[1;1H" % (27)
#         # Decode the bytes as UTF-8 since JSON may only contain UTF-8
#         events = [
#             (
#                 frame_duration,
#                 (clear_code + frame.replace(b"\n", b"\r\n")).decode("utf-8"),
#             )
#             for frame in self.frames
#         ]

#         # Calculate frame size from the largest frames.
#         # Add some padding since we'll get cut off otherwise.
#         height = max(frame.count(b"\n") for frame in self.frames) + 1
#         width = (
#             max(max(len(line) for line in frame.split(b"\n"))
#                 for frame in self.frames)
#             + 2
#         )

#         data = {
#             "version": 1,
#             "width": width,
#             "height": height,
#             "duration": len(self.frames) * frame_duration,
#             "command": "-",
#             "title": "gym VideoRecorder episode",
#             "env": {},  # could add some env metadata here
#             "stdout": events,
#         }

#         with open(self.output_path, "w") as f:
#             json.dump(data, f)

#     @property
#     def version_info(self):
#         return {"backend": "TextEncoder", "version": 1}


# class ImageEncoder:
#     def __init__(self, output_path, frame_shape, frames_per_sec, output_frames_per_sec):
#         self.proc = None
#         self.output_path = output_path
#         # Frame shape should be lines-first, so w and h are swapped
#         h, w, pixfmt = frame_shape
#         if pixfmt != 3 and pixfmt != 4:
#             raise error.InvalidFrame(
#                 "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel.".format(
#                     frame_shape
#                 )
#             )
#         self.wh = (w, h)
#         self.includes_alpha = pixfmt == 4
#         self.frame_shape = frame_shape
#         self.frames_per_sec = frames_per_sec
#         self.output_frames_per_sec = output_frames_per_sec

#         if distutils.spawn.find_executable("avconv") is not None:
#             self.backend = "avconv"
#         elif distutils.spawn.find_executable("ffmpeg") is not None:
#             self.backend = "ffmpeg"
#         elif pkgutil.find_loader("imageio_ffmpeg"):
#             import imageio_ffmpeg

#             self.backend = imageio_ffmpeg.get_ffmpeg_exe()
#         else:
#             raise error.DependencyNotInstalled(
#                 """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`. Alternatively, please install imageio-ffmpeg with `pip install imageio-ffmpeg`"""
#             )

#         self.start()

#     @property
#     def version_info(self):
#         return {
#             "backend": self.backend,
#             "version": str(
#                 subprocess.check_output(
#                     [self.backend, "-version"], stderr=subprocess.STDOUT
#                 )
#             ),
#             "cmdline": self.cmdline,
#         }

#     def start(self):
#         self.cmdline = (
#             self.backend,
#             "-nostats",
#             "-loglevel",
#             "error",  # suppress warnings
#             "-y",
#             # input
#             "-f",
#             "rawvideo",
#             "-s:v",
#             "{}x{}".format(*self.wh),
#             "-pix_fmt",
#             ("rgb32" if self.includes_alpha else "rgb24"),
#             "-framerate",
#             "%d" % self.frames_per_sec,
#             "-i",
#             "-",  # this used to be /dev/stdin, which is not Windows-friendly
#             # output
#             "-vf",
#             "scale=trunc(iw/2)*2:trunc(ih/2)*2",
#             "-vcodec",
#             "libx264",
#             "-pix_fmt",
#             "yuv420p",
#             "-r",
#             "%d" % self.output_frames_per_sec,
#             self.output_path,
#         )

#         logger.debug('Starting %s with "%s"',
#                      self.backend, " ".join(self.cmdline))
#         if hasattr(os, "setsid"):  # setsid not present on Windows
#             self.proc = subprocess.Popen(
#                 self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
#             )
#         else:
#             self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

#     def capture_frame(self, frame):
#         if not isinstance(frame, (np.ndarray, np.generic)):
#             raise error.InvalidFrame(
#                 f"Wrong type {type(frame)} for {frame} (must be np.ndarray or np.generic)"
#             )
#         if frame.shape != self.frame_shape:
#             raise error.InvalidFrame(
#                 f"Your frame has shape {frame.shape}, but the VideoRecorder is configured for shape {self.frame_shape}."
#             )
#         if frame.dtype != np.uint8:
#             raise error.InvalidFrame(
#                 f"Your frame has data type {frame.dtype}, but we require uint8 (i.e. RGB values from 0-255)."
#             )

#         try:
#             if distutils.version.LooseVersion(
#                 np.__version__
#             ) >= distutils.version.LooseVersion("1.9.0"):
#                 self.proc.stdin.write(frame.tobytes())
#             else:
#                 self.proc.stdin.write(frame.tostring())
#         except Exception as e:
#             stdout, stderr = self.proc.communicate()
#             logger.error("VideoRecorder encoder failed: %s", stderr)

#     def close(self):
#         self.proc.stdin.close()
#         ret = self.proc.wait()
#         if ret != 0:
#             logger.error(f"VideoRecorder encoder exited with status {ret}")
