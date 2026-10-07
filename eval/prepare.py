from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SINTEL_TAG = 202021.25
SINTEL_ARCHIVES = (
    (
        "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
        "MPI-Sintel-training_images.zip",
        ("clean",),
    ),
    (
        "http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip",
        "MPI-Sintel-depth-training-20150305.zip",
        ("depth", "camdata_left"),
    ),
)
SINTEL_HF_REPO = "KevinConnorLee/Sintel"
SINTEL_HF_REVISION = "8304a6a05a71c5099eff2c2fb729858c6b018711"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-omega-eval"})
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        temporary.open("wb") as output,
    ):
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    os.replace(temporary, destination)
    return destination


def _hf_file(repo: str, filename: str, destination: Path, revision: str | None) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            local_dir=str(destination),
        )
    )


def _extract_sintel(
    archive: Path, destination: Path, subdirs: tuple[str, ...], scenes: tuple[str, ...]
) -> None:
    targets = {f"training/{subdir}/{scene}/" for subdir in subdirs for scene in scenes}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            normalized = member.filename.replace("\\", "/")
            if member.is_dir() or not any(target in normalized for target in targets):
                continue
            start = normalized.index("training/")
            output = destination / normalized[start:]
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, output.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


def _read_sintel_depth(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if np.fromfile(handle, np.float32, 1)[0] != SINTEL_TAG:
            raise ValueError(path)
        width = int(np.fromfile(handle, np.int32, 1)[0])
        height = int(np.fromfile(handle, np.int32, 1)[0])
        values = np.fromfile(handle, np.float32)
    if values.size != width * height:
        raise ValueError(path)
    return values.reshape(height, width)


def _read_sintel_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        if np.fromfile(handle, np.float32, 1)[0] != SINTEL_TAG:
            raise ValueError(path)
        values = np.fromfile(handle, np.float64, 21)
    if values.size != 21:
        raise ValueError(path)
    return values[:9].reshape(3, 3), values[9:].reshape(3, 4)


def _sintel_raw(args: argparse.Namespace, scenes: tuple[str, ...]) -> tuple[Path, str]:
    if args.raw_root:
        return Path(args.raw_root).expanduser().resolve(), "local"
    work = (
        Path(args.work_dir or Path(args.output).with_name("sintel_downloads"))
        .expanduser()
        .resolve()
    )
    raw = work / "raw"
    if args.source in {"auto", "official", "local"}:
        try:
            archives = (
                Path(args.archive_dir).expanduser().resolve()
                if args.archive_dir
                else work / "archives"
            )
            archives.mkdir(parents=True, exist_ok=True)
            for official_url, mirror_name, subdirs in SINTEL_ARCHIVES:
                official_name = official_url.rsplit("/", 1)[-1]
                archive = next(
                    (
                        path
                        for path in (archives / official_name, archives / mirror_name)
                        if path.is_file()
                    ),
                    None,
                )
                if archive is None:
                    if args.source == "local":
                        raise FileNotFoundError(archives / official_name)
                    archive = _download(official_url, archives / official_name)
                print(f"{archive.name} sha256={_sha256(archive)}")
                _extract_sintel(archive, raw, subdirs, scenes)
            return raw, "official"
        except Exception as error:
            if args.source != "auto":
                raise
            print(f"official Sintel download unavailable: {error}")
    archives = work / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    for _, mirror_name, subdirs in SINTEL_ARCHIVES:
        repo = args.hf_repo or SINTEL_HF_REPO
        revision = args.hf_revision or (
            SINTEL_HF_REVISION if repo == SINTEL_HF_REPO else None
        )
        archive = _hf_file(repo, mirror_name, archives, revision)
        print(f"{archive.name} sha256={_sha256(archive)}")
        _extract_sintel(archive, raw, subdirs, scenes)
    return raw, "huggingface"


def _prepare_sintel(args: argparse.Namespace) -> None:
    frame_map = json.loads((ROOT / "frames" / "sintel.json").read_text())
    scenes = tuple(args.scenes or frame_map)
    raw_root, source = _sintel_raw(args, scenes)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene in scenes:
        image_root = raw_root / "training" / "clean" / scene
        depth_root = raw_root / "training" / "depth" / scene
        camera_root = raw_root / "training" / "camdata_left" / scene
        images = sorted(image_root.glob("frame_*.png"))
        selected = set(frame_map[scene])
        scene_output = output_root / scene
        (scene_output / "images").mkdir(parents=True, exist_ok=True)
        (scene_output / "depths").mkdir(parents=True, exist_ok=True)
        names, extrinsics, intrinsics = [], [], []
        for index, image_path in enumerate(images):
            camera_path = camera_root / f"frame_{index + 1:04d}.cam"
            depth_path = depth_root / f"frame_{index + 1:04d}.dpt"
            intrinsic, extrinsic = _read_sintel_camera(camera_path)
            names.append(f"{index:05d}.png")
            height, width = 360, 846
            scaled = intrinsic.copy()
            scaled[0] *= width / 1024
            scaled[1] *= height / 436
            intrinsics.append(scaled)
            extrinsics.append(extrinsic)
            if index not in selected:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            resized = cv2.resize(
                image.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                str(scene_output / "images" / f"{index:05d}.png"),
                np.clip(resized, 0, 255).astype(np.uint8),
            )
            depth = cv2.resize(
                _read_sintel_depth(depth_path),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            np.savez_compressed(
                scene_output / "depths" / f"{index:05d}.npz", depth=depth
            )
            total += 1
        (scene_output / "frames.txt").write_text("\n".join(names) + "\n")
        np.savez_compressed(
            scene_output / "cameras.npz",
            extrinsics=np.asarray(extrinsics),
            intrinsics=np.asarray(intrinsics),
        )
        print(f"{scene}: {len(selected)} frames")
    (output_root / "source.json").write_text(
        json.dumps(
            {
                "dataset": "MPI-Sintel",
                "frames": total,
                "location": (
                    (args.hf_repo or SINTEL_HF_REPO)
                    if source == "huggingface"
                    else (
                        [item[0] for item in SINTEL_ARCHIVES]
                        if source == "official"
                        else "local"
                    )
                ),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "revision": (
                    args.hf_revision
                    or (
                        SINTEL_HF_REVISION
                        if (args.hf_repo or SINTEL_HF_REPO) == SINTEL_HF_REPO
                        else None
                    )
                )
                if source == "huggingface"
                else None,
                "source": source,
            },
            indent=2,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    command = subparsers.add_parser("sintel")
    command.add_argument("--output", required=True)
    command.add_argument("--source", choices=("auto", "official", "hf", "local"), default="auto")
    command.add_argument("--raw-root")
    command.add_argument("--archive-dir")
    command.add_argument("--work-dir")
    command.add_argument("--hf-repo")
    command.add_argument("--hf-revision")
    command.add_argument("--scenes", nargs="+")
    return parser.parse_args()


def main() -> int:
    _prepare_sintel(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
