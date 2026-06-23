"""Face detection, clustering, and image sorting engine."""

from __future__ import annotations

import csv
import math
import os
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Protocol

import cv2
import numpy as np

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif",
)
MODEL_DIR = Path(__file__).with_name("models")
YU_NET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YU_NET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
KNOWN_MATCH_THRESHOLD = 0.45
AMBIGUITY_MARGIN = 0.03
CLUSTER_THRESHOLD = 0.5
DETECTION_SCORE_THRESHOLD = 0.9
MIN_FACE_SIZE = 80
MAX_IMAGE_DIM = 1600
DEFAULT_WORKERS = min(8, max(2, (os.cpu_count() or 4)))

_thread_local = threading.local()


class ProgressCallback(Protocol):
    def __call__(
        self,
        phase: str,
        current: int,
        total: int,
        message: str,
        detail: str = "",
    ) -> None: ...


def _noop_progress(*_args: object, **_kwargs: object) -> None:
    pass


@dataclass
class DetectedFace:
    feature: np.ndarray
    bbox: tuple[int, int, int, int]
    detector_score: float


@dataclass
class PersonReference:
    name: str
    features: list[np.ndarray]

    @property
    def centroid(self) -> np.ndarray:
        vectors = np.vstack(self.features)
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid if norm == 0 else centroid / norm


@dataclass
class FaceCluster:
    label: str
    features: list[np.ndarray]

    @property
    def centroid(self) -> np.ndarray:
        vectors = np.vstack(self.features)
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid if norm == 0 else centroid / norm


@dataclass
class SortConfig:
    scan_folder: Path
    output_folder: Path
    reference_folder: Path | None = None
    report_path: Path | None = None
    auto_cluster: bool = True
    workers: int = DEFAULT_WORKERS
    cluster_threshold: float = CLUSTER_THRESHOLD


@dataclass
class SortResult:
    image_count: int
    face_count: int
    group_count: int
    images_without_faces: int
    report_path: Path


def ensure_model(path: Path, url: str, progress: ProgressCallback = _noop_progress) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_download = True
    if path.exists():
        with path.open("rb") as model_file:
            header = model_file.read(32)
        needs_download = header.startswith(b"version https://git-lfs")

    if needs_download:
        progress("models", 0, 1, f"Downloading {path.name}...")
        urllib.request.urlretrieve(url, path)
    return path


def build_detector() -> cv2.FaceDetectorYN:
    model_path = ensure_model(YU_NET_PATH, YU_NET_URL)
    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        DETECTION_SCORE_THRESHOLD,
        0.3,
        5000,
    )


def build_recognizer() -> cv2.FaceRecognizerSF:
    model_path = ensure_model(SFACE_PATH, SFACE_URL)
    return cv2.FaceRecognizerSF.create(str(model_path), "")


def _get_thread_models() -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = build_detector()
        _thread_local.recognizer = build_recognizer()
    return _thread_local.detector, _thread_local.recognizer


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", label.strip())
    return cleaned.rstrip(" .") or "unnamed"


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _ensure_heif_support() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def diagnose_empty_scan(folder: Path, exclude_roots: tuple[Path, ...] = ()) -> str:
    if not folder.exists():
        return f"The folder does not exist: {folder}"

    all_files = [path for path in folder.rglob("*") if path.is_file()]
    if not all_files:
        return (
            f"No files were found in:\n{folder}\n\n"
            "Check that you selected the folder that actually contains your photos."
        )

    extension_counts: dict[str, int] = {}
    for path in all_files:
        ext = path.suffix.lower() or "(no extension)"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    supported = [
        path
        for path in all_files
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    excluded = [
        path
        for path in supported
        if any(is_within(path, exclude_root) for exclude_root in exclude_roots)
    ]
    usable = [
        path
        for path in supported
        if not any(is_within(path, exclude_root) for exclude_root in exclude_roots)
    ]

    lines = [
        f"Scanned folder: {folder}",
        f"Total files found: {len(all_files)}",
        "File types present: "
        + ", ".join(f"{ext} ({count})" for ext, count in sorted(extension_counts.items())),
    ]

    if exclude_roots:
        for exclude_root in exclude_roots:
            lines.append(f"Excluded path (output or reference inside source): {exclude_root}")
        if excluded:
            lines.append(
                f"{len(excluded)} image(s) were skipped because they sit inside an excluded folder."
            )

    unsupported = len(all_files) - len(supported)
    if unsupported and not usable:
        lines.append("")
        lines.append("None of the files use a supported image format.")
        if extension_counts.get(".heic") or extension_counts.get(".heif"):
            lines.append(
                "HEIC/HEIF detected — run: pip install pillow pillow-heif"
            )
        else:
            lines.append(f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}")

    if usable:
        lines.append(f"Usable images after exclusions: {len(usable)}")

    return "\n".join(lines)


def list_images(folder: Path, exclude_roots: tuple[Path, ...] = ()) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(is_within(path, exclude_root) for exclude_root in exclude_roots)
    )


def face_area(face: np.ndarray) -> float:
    return float(face[2] * face[3])


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    flattened = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(flattened)
    return flattened if norm == 0 else flattened / norm


def _resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _load_via_pillow(image_path: Path) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(image_path) as pil_image:
            rgb = pil_image.convert("RGB")
            array = np.asarray(rgb)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray | None:
    suffix = image_path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        _ensure_heif_support()
        image = _load_via_pillow(image_path)
        if image is not None:
            return _resize_image(image, max_dim)

    try:
        file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None

    if file_bytes.size == 0:
        return None

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None and suffix in {".tif", ".tiff", ".gif", ".webp"}:
        image = _load_via_pillow(image_path)
    if image is None:
        return None

    return _resize_image(image, max_dim)


def detect_faces_in_image(
    image_path: Path,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
) -> list[DetectedFace]:
    image = load_image(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None:
        return []

    sorted_faces = sorted(faces, key=face_area, reverse=True)
    detected_faces: list[DetectedFace] = []

    for face in sorted_faces:
        x, y, w, h = [int(round(value)) for value in face[:4]]
        score = float(face[-1])
        if min(w, h) < MIN_FACE_SIZE:
            continue
        aligned_face = recognizer.alignCrop(image, face)
        feature = normalize_feature(recognizer.feature(aligned_face))
        detected_faces.append(
            DetectedFace(feature=feature, bbox=(x, y, w, h), detector_score=score)
        )

    return detected_faces


def _detect_worker(image_path: Path) -> tuple[Path, list[DetectedFace]]:
    detector, recognizer = _get_thread_models()
    faces = detect_faces_in_image(image_path, detector, recognizer)
    return image_path, faces


def detect_faces_parallel(
    image_paths: list[Path],
    workers: int,
    progress: ProgressCallback,
    cancel_event: threading.Event | None = None,
) -> dict[Path, list[DetectedFace]]:
    results: dict[Path, list[DetectedFace]] = {}
    total = len(image_paths)
    if total == 0:
        return results

    completed = 0
    worker_count = min(workers, total)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_detect_worker, path): path for path in image_paths}
        for future in as_completed(futures):
            if cancel_event and cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            image_path, faces = future.result()
            results[image_path] = faces
            completed += 1
            face_note = f"{len(faces)} face(s)" if faces else "no faces"
            progress(
                "detect",
                completed,
                total,
                f"Scanned {image_path.name}",
                f"{face_note} in {image_path.name}",
            )

    return results


def pick_reference_face(faces: list[DetectedFace], image_path: Path) -> DetectedFace | None:
    if not faces:
        return None
    return faces[0]


def load_known_faces(
    reference_root: Path,
    face_map: dict[Path, list[DetectedFace]] | None,
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    progress: ProgressCallback = _noop_progress,
) -> list[PersonReference]:
    if not reference_root.exists():
        raise FileNotFoundError(f"Reference folder does not exist: {reference_root}")

    person_references: list[PersonReference] = []
    subfolders = sorted(path for path in reference_root.iterdir() if path.is_dir())

    if subfolders:
        for index, person_folder in enumerate(subfolders, start=1):
            progress(
                "references",
                index,
                len(subfolders),
                f"Loading reference: {person_folder.name}",
            )
            features: list[np.ndarray] = []
            for image_path in list_images(person_folder):
                faces = (
                    face_map.get(image_path)
                    if face_map is not None
                    else detect_faces_in_image(image_path, detector, recognizer)
                )
                if faces is None:
                    faces = []
                face = pick_reference_face(faces, image_path)
                if face:
                    features.append(face.feature)
            if features:
                person_references.append(
                    PersonReference(name=sanitize_label(person_folder.name), features=features)
                )
    else:
        ref_images = list_images(reference_root)
        for index, image_path in enumerate(ref_images, start=1):
            progress(
                "references",
                index,
                len(ref_images),
                f"Loading reference: {image_path.name}",
            )
            faces = (
                face_map.get(image_path)
                if face_map is not None
                else detect_faces_in_image(image_path, detector, recognizer)
            )
            if faces is None:
                faces = []
            face = pick_reference_face(faces, image_path)
            if face:
                person_references.append(
                    PersonReference(
                        name=sanitize_label(image_path.stem),
                        features=[face.feature],
                    )
                )

    if not person_references:
        raise RuntimeError("No usable reference faces were found.")

    return person_references


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def score_person(face_feature: np.ndarray, person: PersonReference) -> tuple[float, float]:
    centroid_score = cosine_similarity(face_feature, person.centroid)
    sample_scores = [cosine_similarity(face_feature, feature) for feature in person.features]
    top_sample_score = max(sample_scores)
    return centroid_score, top_sample_score


def match_known_face(
    face_feature: np.ndarray, references: list[PersonReference]
) -> tuple[str | None, float, str]:
    scored_matches: list[tuple[float, float, str]] = []

    for person in references:
        centroid_score, sample_score = score_person(face_feature, person)
        combined_score = (centroid_score * 0.7) + (sample_score * 0.3)
        scored_matches.append((combined_score, sample_score, person.name))

    scored_matches.sort(reverse=True)
    best_score, best_sample_score, best_name = scored_matches[0]
    second_score = scored_matches[1][0] if len(scored_matches) > 1 else -1.0

    if best_score < KNOWN_MATCH_THRESHOLD:
        return None, best_score, "below_threshold"
    if best_sample_score < KNOWN_MATCH_THRESHOLD:
        return None, best_score, "weak_reference_support"
    if best_score - second_score < AMBIGUITY_MARGIN:
        return None, best_score, "ambiguous"
    return best_name, best_score, "matched"


def assign_cluster(
    face_feature: np.ndarray,
    clusters: list[FaceCluster],
    threshold: float,
    label_prefix: str = "person",
) -> tuple[str, float, str]:
    best_cluster: FaceCluster | None = None
    best_score = -1.0

    for cluster in clusters:
        score = cosine_similarity(face_feature, cluster.centroid)
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is not None and best_score >= threshold:
        best_cluster.features.append(face_feature)
        return best_cluster.label, best_score, f"matched {best_cluster.label} ({best_score:.2f})"

    label = f"{label_prefix}_{len(clusters) + 1:03d}"
    clusters.append(FaceCluster(label=label, features=[face_feature]))
    return label, 1.0, f"new group {label}"


def relative_output_path(output_folder: Path, person_name: str, image_path: Path) -> Path:
    return output_folder / person_name / image_path.name


def copy_once(image_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    copy2(image_path, destination)


def write_report(
    report_path: Path,
    rows: list[dict[str, str | int | float]],
    reference_count: int,
    cluster_count: int,
    images_without_faces: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        report_file.write(f"# known_people,{reference_count}\n")
        report_file.write(f"# face_groups,{cluster_count}\n")
        report_file.write(f"# images_without_faces,{images_without_faces}\n")
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "image_path",
                "face_index",
                "label",
                "status",
                "match_score",
                "detector_score",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def sort_images(
    config: SortConfig,
    progress: ProgressCallback = _noop_progress,
    cancel_event: threading.Event | None = None,
) -> SortResult:
    scan_folder = config.scan_folder.expanduser().resolve()
    output_folder = config.output_folder.expanduser().resolve()
    reference_folder = (
        config.reference_folder.expanduser().resolve()
        if config.reference_folder
        else None
    )
    report_path = (
        config.report_path.expanduser().resolve()
        if config.report_path
        else output_folder / "face_report.csv"
    )

    progress("models", 0, 1, "Loading face recognition models...")
    build_detector()
    build_recognizer()

    if scan_folder.resolve() == output_folder.resolve():
        raise ValueError(
            "Source and output folders must be different.\n"
            "Choose a source folder with your photos and a separate output folder "
            "where sorted copies will be placed."
        )

    exclude_roots: list[Path] = []
    if is_within(output_folder, scan_folder):
        exclude_roots.append(output_folder)
    if reference_folder and reference_folder != scan_folder and is_within(reference_folder, scan_folder):
        exclude_roots.append(reference_folder)

    progress("scan", 0, 1, f"Finding images in {scan_folder}...")
    image_paths = list_images(scan_folder, tuple(exclude_roots))
    progress("scan", 1, 1, f"Found {len(image_paths)} images in {scan_folder.name}")

    if not image_paths:
        raise ValueError(
            "No images were found to sort.\n\n" + diagnose_empty_scan(scan_folder, tuple(exclude_roots))
        )

    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Sorting cancelled.")

    face_map = detect_faces_parallel(
        image_paths,
        config.workers,
        progress,
        cancel_event,
    )

    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Sorting cancelled.")

    references: list[PersonReference] = []
    use_references = reference_folder is not None and not config.auto_cluster

    if use_references:
        progress("references", 0, 1, "Building reference profiles...")
        detector, recognizer = _get_thread_models()
        references = load_known_faces(
            reference_folder,
            face_map,
            detector,
            recognizer,
            progress,
        )
        progress("references", 1, 1, f"Loaded {len(references)} known people")

    output_folder.mkdir(parents=True, exist_ok=True)
    clusters: list[FaceCluster] = []
    report_rows: list[dict[str, str | int | float]] = []
    images_without_faces = 0
    total_faces = 0
    assignments: dict[Path, set[str]] = {}

    progress("match", 0, len(image_paths), "Grouping faces...")
    for index, image_path in enumerate(image_paths, start=1):
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Sorting cancelled.")

        faces = face_map.get(image_path, [])
        if not faces:
            images_without_faces += 1
            progress(
                "match",
                index,
                len(image_paths),
                f"No face in {image_path.name}",
                "skipped — no detectable face",
            )
            continue

        assigned_labels: set[str] = set()

        for face_index, face in enumerate(faces, start=1):
            total_faces += 1

            if use_references:
                person_name, match_score, match_reason = match_known_face(
                    face.feature, references
                )
                if person_name is None:
                    person_name, cluster_score, detail = assign_cluster(
                        face.feature,
                        clusters,
                        config.cluster_threshold,
                        label_prefix="unknown_group",
                    )
                    match_reason = f"unknown:{match_reason}"
                    match_score = cluster_score if math.isclose(match_score, -1.0) else match_score
                else:
                    detail = f"matched {person_name} ({match_score:.2f})"
            else:
                person_name, match_score, detail = assign_cluster(
                    face.feature,
                    clusters,
                    config.cluster_threshold,
                )
                match_reason = "clustered"

            assigned_labels.add(person_name)
            progress(
                "match",
                index,
                len(image_paths),
                f"Compared {image_path.name}",
                detail,
            )
            report_rows.append(
                {
                    "image_path": str(image_path),
                    "face_index": face_index,
                    "label": person_name,
                    "status": match_reason,
                    "match_score": round(match_score, 4),
                    "detector_score": round(face.detector_score, 4),
                    "bbox_x": face.bbox[0],
                    "bbox_y": face.bbox[1],
                    "bbox_w": face.bbox[2],
                    "bbox_h": face.bbox[3],
                }
            )

        assignments[image_path] = assigned_labels

    copy_total = sum(len(labels) for labels in assignments.values())
    copied = 0
    for image_path, labels in assignments.items():
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Sorting cancelled.")
        for label in labels:
            copied += 1
            destination = relative_output_path(output_folder, label, image_path)
            copy_once(image_path, destination)
            progress(
                "copy",
                copied,
                max(copy_total, 1),
                f"Copied to {label}/",
                f"{image_path.name} → {label}/",
            )

    group_count = len(clusters) if clusters else len(references)
    if not use_references:
        group_count = len(clusters)
    elif clusters:
        group_count = len(references) + len(clusters)

    write_report(
        report_path,
        report_rows,
        len(references),
        len(clusters),
        images_without_faces,
    )

    progress(
        "done",
        len(image_paths),
        len(image_paths),
        "Sorting complete",
        (
            f"{len(image_paths)} images, {total_faces} faces, "
            f"{group_count} groups, report at {report_path.name}"
        ),
    )

    return SortResult(
        image_count=len(image_paths),
        face_count=total_faces,
        group_count=group_count,
        images_without_faces=images_without_faces,
        report_path=report_path,
    )
