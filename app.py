import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from scenedetect import open_video, detect
from scenedetect.detectors import ThresholdDetector, AdaptiveDetector
from scenedetect.scene_manager import save_images
from scenedetect.frame_timecode import FrameTimecode


@dataclass
class Settings:
    storage_account: str
    container: str
    shortcode: str

    input_dir: Path
    output_dir: Path

    threshold: float
    min_scene_len: int
    default_images_per_scene: int


class VideoSplitter:
    def __init__(self, video_path: Path, threshold: float = 27.0):
        self.video_path = video_path
        self.video = open_video(str(video_path))

        detector = AdaptiveDetector(
            # threshold=threshold,
            # add_final_scene=True,
        )
        scene_list = detect(str(video_path), detector)
        self.scene_list = self._ensure_at_least_one_scene(scene_list)

    def _ensure_at_least_one_scene(self, scene_list):
        if scene_list:
            return scene_list

        start = FrameTimecode(0, self.video.frame_rate)
        if hasattr(self.video, "duration") and self.video.duration:
            end = self.video.duration
        else:
            total_frames = getattr(self.video, "_frame_count", 0)
            end = FrameTimecode(total_frames, self.video.frame_rate)
        return [(start, end)]

    def _num_images_for_scenes(self, default_per_scene: int) -> int:
        if len(self.scene_list) == 1:
            start, end = self.scene_list[0]
            duration_seconds = max(1, math.ceil(end.get_seconds() - start.get_seconds()))
            return duration_seconds
        return default_per_scene

    def save_thumbnails(self, thumbnails_dir: Path, default_images_per_scene: int):
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        num_images = self._num_images_for_scenes(default_images_per_scene)

        save_images(
            scene_list=self.scene_list,
            video=self.video,
            output_dir=str(thumbnails_dir),
            image_name_template="scene-$SCENE_NUMBER-image-$IMAGE_NUMBER",
            image_extension="jpg",
            num_images=num_images,
        )

    def get_scenes(self):
        return self.scene_list


def load_settings() -> Settings:
    shortcode = os.getenv("SHORTCODE")
    if not shortcode:
        raise ValueError("SHORTCODE env var is required (instagram shortcode folder name)")

    return Settings(
        storage_account=os.getenv("STORAGE_ACCOUNT", "socialshopper"),
        container=os.getenv("CONTAINER", "downloads"),
        shortcode=shortcode,
        input_dir=Path(os.getenv("INPUT_DIR", "/data/in")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "/data/out")),
        threshold=float(os.getenv("THRESHOLD", "27.0")),
        min_scene_len=int(os.getenv("MIN_SCENE_LEN", "15")),
        default_images_per_scene=int(os.getenv("DEFAULT_IMAGES_PER_SCENE", "3")),
    )


def blob_service_client(settings: Settings) -> BlobServiceClient:
    cred = DefaultAzureCredential()
    return BlobServiceClient(
        account_url=f"https://{settings.storage_account}.blob.core.windows.net",
        credential=cred,
    )


def list_mp4_blobs(bsc: BlobServiceClient, container: str, prefix: str) -> List[str]:
    cc = bsc.get_container_client(container)
    mp4s = []
    for blob in cc.list_blobs(name_starts_with=prefix):
        if blob.name.lower().endswith(".mp4"):
            mp4s.append(blob.name)
    return sorted(mp4s)


def download_blob_to_file(bsc: BlobServiceClient, container: str, blob_name: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    bc = bsc.get_blob_client(container=container, blob=blob_name)
    with dst.open("wb") as f:
        stream = bc.download_blob()
        stream.readinto(f)


def upload_file(
    bsc: BlobServiceClient,
    container: str,
    blob_name: str,
    src: Path,
    content_type: str | None = None,
):
    bc = bsc.get_blob_client(container=container, blob=blob_name)
    cs = ContentSettings(content_type=content_type) if content_type else None
    with src.open("rb") as f:
        bc.upload_blob(f, overwrite=True, content_settings=cs)


def main():
    s = load_settings()
    s.input_dir.mkdir(parents=True, exist_ok=True)
    s.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{s.shortcode}/"
    out_prefix = f"{s.shortcode}/output_scenes/"

    bsc = blob_service_client(s)

    # 1) Find the single mp4 under the shortcode prefix
    mp4s = list_mp4_blobs(bsc, s.container, prefix)
    if len(mp4s) == 0:
        raise FileNotFoundError(f"No mp4 blobs found under {s.container}/{prefix}")
    if len(mp4s) > 1:
        raise RuntimeError(f"Expected exactly one mp4 under {s.container}/{prefix}, found: {mp4s}")

    mp4_blob_name = mp4s[0]

    # 2) Download to local disk
    local_video = s.input_dir / "video.mp4"
    print(f"[sdk] downloading: {mp4_blob_name} -> {local_video}")
    download_blob_to_file(bsc, s.container, mp4_blob_name, local_video)

    # 3) Process locally
    splitter = VideoSplitter(local_video, threshold=s.threshold)
    scenes = splitter.get_scenes()

    thumbnails_dir = s.output_dir / "thumbnails"
    splitter.save_thumbnails(thumbnails_dir, default_images_per_scene=s.default_images_per_scene)

    manifest = {
        "shortcode": s.shortcode,
        "source_blob": mp4_blob_name,
        "threshold": s.threshold,
        "default_images_per_scene": s.default_images_per_scene,
        "num_scenes": len(scenes),
        "scenes": [
            {"index": i + 1, "start": sc[0].get_timecode(), "end": sc[1].get_timecode()}
            for i, sc in enumerate(scenes)
        ],
        "thumbnails": [p.name for p in sorted(thumbnails_dir.glob("*.jpg"))],
    }

    manifest_path = s.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4) Upload outputs back to blob
    print(f"[sdk] uploading results to: {s.container}/{out_prefix}")

    upload_file(bsc, s.container, out_prefix + "manifest.json", manifest_path, content_type="application/json")

    for jpg in sorted(thumbnails_dir.glob("*.jpg")):
        upload_file(
            bsc,
            s.container,
            out_prefix + f"thumbnails/{jpg.name}",
            jpg,
            content_type="image/jpeg",
        )

    print("[sdk] done")


if __name__ == "__main__":
    main()
