import os
from scenedetect import open_video, detect
from scenedetect.detectors import ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images
from scenedetect.frame_timecode import FrameTimecode
import math

path = "../video-files/99def53c-a7cb-4d6e-be55-267e0e374b56.mp4"

class VideoSplitter:
    def __init__(self, video_path, threshold=27.0, min_scene_len=15):
        self.video_path = video_path
        self.video = open_video(video_path)

        detector = ThresholdDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            add_final_scene=True,
        )

        scene_list = detect(video_path, detector)
        self.scene_list = self._ensure_at_least_one_scene(scene_list)

    def _ensure_at_least_one_scene(self, scene_list):
        if scene_list:
            return scene_list

        start = FrameTimecode(0, self.video.frame_rate)

        # Get total frames in a backend-safe way
        if hasattr(self.video, "duration") and self.video.duration:
            end = self.video.duration
        else:
            # VideoStreamCv2 fallback
            total_frames = getattr(self.video, "_frame_count", 0)
            end = FrameTimecode(total_frames, self.video.frame_rate)

        return [(start, end)]

    def _num_images_for_scenes(self, default_per_scene=3):
        # If only one scene, sample 1 image per second across the whole video
        if len(self.scene_list) == 1:
            start, end = self.scene_list[0]
            duration_seconds = max(1, math.ceil(end.get_seconds() - start.get_seconds()))
            return duration_seconds

        # Otherwise, keep a fixed number per scene
        return default_per_scene

    def split_video(self, output_dir, save_clips=False, save_scene_images=True, default_images_per_scene=3):
        os.makedirs(output_dir, exist_ok=True)

        if save_clips:
            split_video_ffmpeg(self.video_path, self.scene_list, output_dir=output_dir)

        if save_scene_images:
            images_dir = os.path.join(output_dir, "thumbnails")
            os.makedirs(images_dir, exist_ok=True)

            num_images = self._num_images_for_scenes(default_per_scene=default_images_per_scene)

            save_images(
                scene_list=self.scene_list,
                video=self.video,
                output_dir=images_dir,
                image_name_template="scene-$SCENE_NUMBER-image-$IMAGE_NUMBER",
                image_extension="jpg",
                num_images=num_images,
            )

    def get_scenes(self):
        return self.scene_list


if __name__ == "__main__":
    video_splitter = VideoSplitter(path)
    scenes = video_splitter.get_scenes()

    print(f"Detected {len(scenes)} scenes.")
    for i, scene in enumerate(scenes):
        print(f"Scene {i + 1}: Start - {scene[0].get_timecode()}, End - {scene[1].get_timecode()}")

    output_directory = f"./output_scenes/{os.path.splitext(os.path.basename(path))[0]}/thumbnails"
    video_splitter.split_video(output_directory, save_clips=False, save_scene_images=True)
    print(f"Saved outputs in '{output_directory}'.")