#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]
DEFAULT_PLATE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "license_plate_detector.pt"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class PlateDetection:
    box: tuple[int, int, int, int]
    text: str
    confidence: float


@dataclass
class CrossingEvent:
    frame: int
    time_seconds: float
    track_id: int
    class_name: str
    center_x: int
    center_y: int
    plate_text: str


@dataclass
class CountingLine:
    start: Point
    end: Point
    direction: str

    def is_inside_frame(self, frame_width: int, frame_height: int) -> bool:
        return self._point_inside_frame(self.start, frame_width, frame_height) and self._point_inside_frame(
            self.end, frame_width, frame_height
        )

    def crosses(self, previous: Point, current: Point) -> bool:
        previous_side = self._signed_side(previous)
        current_side = self._signed_side(current)
        changed_side = previous_side == 0 or current_side == 0 or previous_side * current_side < 0
        return changed_side and self._movement_matches_direction(previous, current)

    def draw(self, frame: np.ndarray) -> None:
        cv2.line(
            frame,
            (int(self.start.x), int(self.start.y)),
            (int(self.end.x), int(self.end.y)),
            (0, 0, 255),
            3,
        )

    def _signed_side(self, point: Point) -> float:
        return (self.end.x - self.start.x) * (point.y - self.start.y) - (
            self.end.y - self.start.y
        ) * (point.x - self.start.x)

    def _movement_matches_direction(self, previous: Point, current: Point) -> bool:
        dx = current.x - previous.x
        dy = current.y - previous.y

        if self.direction == "left-to-right":
            return dx > 0
        if self.direction == "right-to-left":
            return dx < 0
        if self.direction == "top-to-bottom":
            return dy > 0
        if self.direction == "bottom-to-top":
            return dy < 0
        raise ValueError(f"Unsupported direction: {self.direction}")

    @staticmethod
    def _point_inside_frame(point: Point, frame_width: int, frame_height: int) -> bool:
        return 0 <= point.x <= frame_width and 0 <= point.y <= frame_height


@dataclass
class VehicleTrack:
    track_id: int
    class_name: str
    previous_center: Point | None = None
    center: Point | None = None
    counted: bool = False
    plate_text: str = ""
    plate_detections: list[PlateDetection] = field(default_factory=list)

    @property
    def box_color(self) -> tuple[int, int, int]:
        return (0, 180, 0) if self.counted else (255, 120, 0)

    def update(self, center: Point, plate_detections: list[PlateDetection]) -> None:
        self.previous_center = self.center
        self.center = center
        self.plate_detections = plate_detections

        best_text = max(
            (plate for plate in plate_detections if plate.text),
            key=lambda plate: plate.confidence,
            default=None,
        )
        if best_text is not None:
            self.plate_text = best_text.text

    def crosses_line(self, line: CountingLine) -> bool:
        if self.previous_center is None or self.center is None or self.counted:
            return False
        return line.crosses(self.previous_center, self.center)


class LicensePlateRecognizer:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"License plate model was not found: {model_path}. "
                "Expected file: models/license_plate_detector.pt"
            )

        self.detector = YOLO(str(model_path))
        self.reader = easyocr.Reader(["en"], gpu=False)

    def detect_and_read(self, frame: np.ndarray) -> list[PlateDetection]:
        if frame.size == 0:
            return []

        results = self.detector.predict(frame, verbose=False, conf=0.10)
        detections: list[PlateDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = clamp_box(box.xyxy[0].tolist(), frame.shape[1], frame.shape[0])
                plate_crop = frame[y1:y2, x1:x2]
                text, confidence = self._read_plate_text(plate_crop)
                detections.append(
                    PlateDetection(
                        box=(x1, y1, x2, y2),
                        text=text,
                        confidence=confidence,
                    )
                )

        return detections

    def _read_plate_text(self, plate_crop: np.ndarray) -> tuple[str, float]:
        if plate_crop.size == 0:
            return "", 0.0

        best_text = ""
        best_confidence = 0.0
        for variant in self._ocr_variants(plate_crop):
            ocr_results = self.reader.readtext(
                variant,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
            )
            for _, text, confidence in ocr_results:
                normalized = normalize_plate_text(text)
                if normalized and confidence > best_confidence:
                    best_text = normalized
                    best_confidence = float(confidence)

        return best_text, best_confidence

    @staticmethod
    def _ocr_variants(plate_crop: np.ndarray) -> list[np.ndarray]:
        scaled = cv2.resize(plate_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        return [scaled, gray, threshold]


def normalize_plate_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def parse_point(raw: str) -> Point:
    try:
        x_raw, y_raw = raw.split(",", maxsplit=1)
        return Point(float(x_raw), float(y_raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point must be formatted as x,y") from exc


def parse_classes(raw: str) -> list[int]:
    try:
        return [int(class_id.strip()) for class_id in raw.split(",") if class_id.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("classes must be comma-separated COCO class ids") from exc


def clamp_box(
    box: Iterable[float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(x1, frame_width - 1)),
        max(0, min(y1, frame_height - 1)),
        max(0, min(x2, frame_width - 1)),
        max(0, min(y2, frame_height - 1)),
    )


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    y_top = max(0, y - text_h - 8)
    cv2.rectangle(frame, (x, y_top), (x + text_w + 8, y_top + text_h + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y_top + text_h + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_vehicle(
    frame: np.ndarray,
    vehicle: VehicleTrack,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    color = vehicle.box_color
    label = f"ID {vehicle.track_id} {vehicle.class_name}"
    if vehicle.plate_text:
        label += f" plate:{vehicle.plate_text}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if vehicle.center is not None:
        cv2.circle(frame, (int(vehicle.center.x), int(vehicle.center.y)), 4, (0, 255, 255), -1)
    draw_label(frame, label, x1, max(20, y1), color)


def draw_plate_detection(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
) -> None:
    x1, y1, x2, y2 = box
    color = (255, 0, 255)
    label = f"plate {text}" if text else "plate"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_label(frame, label, x1, max(20, y1), color)


def plate_belongs_to_vehicle(
    plate: PlateDetection,
    vehicle_box: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = vehicle_box
    px1, py1, px2, py2 = plate.box
    plate_center = Point((px1 + px2) / 2, (py1 + py2) / 2)
    return x1 <= plate_center.x <= x2 and y1 <= plate_center.y <= y2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, track, count vehicles, and recognize license plates in a video."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument("--output", default="output/output_counted.mp4")
    parser.add_argument("--csv", default="output/crossings.csv")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 vehicle model name or path.")
    parser.add_argument("--plate-model", default=str(DEFAULT_PLATE_MODEL_PATH))
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=0.35, help="Vehicle detection confidence.")
    parser.add_argument("--line-start", type=parse_point, required=True, help="Line point x,y.")
    parser.add_argument("--line-end", type=parse_point, required=True, help="Line point x,y.")
    parser.add_argument(
        "--direction",
        choices=("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"),
        default="top-to-bottom",
        help="Only crossings moving in this direction are counted.",
    )
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=DEFAULT_VEHICLE_CLASSES,
        help="Comma-separated COCO class ids. Default: 2,3,5,7 (car,motorcycle,bus,truck).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="FPS used for processing and output video. Lower FPS reduces tracker/OCR load.",
    )
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    return parser


def should_process_frame(frame_index: int, original_fps: float, target_fps: float) -> bool:
    if target_fps <= 0 or target_fps >= original_fps:
        return True
    frame_step = max(1, round(original_fps / target_fps))
    return (frame_index - 1) % frame_step == 0


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    plate_model_path = Path(args.plate_model)
    counting_line = CountingLine(args.line_start, args.line_end, args.direction)

    if not input_path.exists():
        print(f"Input video does not exist: {input_path}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    vehicle_model = YOLO(args.model)
    plate_recognizer = LicensePlateRecognizer(plate_model_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        print(f"Cannot open video: {input_path}", file=sys.stderr)
        return 2

    original_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    output_fps = min(args.target_fps, original_fps) if args.target_fps > 0 else original_fps
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not counting_line.is_inside_frame(frame_width, frame_height):
        print(
            "Counting line is outside the video frame. "
            f"Video size is {frame_width}x{frame_height}, "
            f"line is ({counting_line.start.x:g},{counting_line.start.y:g}) -> "
            f"({counting_line.end.x:g},{counting_line.end.y:g}).",
            file=sys.stderr,
        )
        return 2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )

    vehicles: dict[int, VehicleTrack] = {}
    events: list[CrossingEvent] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            if not should_process_frame(frame_index, original_fps, args.target_fps):
                continue

            results = vehicle_model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=args.classes,
                conf=args.conf,
                verbose=False,
            )
            result = results[0] if results else None
            frame_plate_detections = plate_recognizer.detect_and_read(frame)

            if result is not None and result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
                    center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                    class_name = vehicle_model.names.get(class_id, str(class_id))
                    vehicle = vehicles.setdefault(track_id, VehicleTrack(track_id, class_name))
                    vehicle.class_name = class_name

                    vehicle_box = (x1, y1, x2, y2)
                    plate_detections = [
                        plate for plate in frame_plate_detections if plate_belongs_to_vehicle(plate, vehicle_box)
                    ]
                    vehicle.update(center, plate_detections)

                    if vehicle.crosses_line(counting_line):
                        vehicle.counted = True
                        events.append(
                            CrossingEvent(
                                frame=frame_index,
                                time_seconds=frame_index / original_fps,
                                track_id=track_id,
                                class_name=class_name,
                                center_x=int(center.x),
                                center_y=int(center.y),
                                plate_text=vehicle.plate_text,
                            )
                        )

                    draw_vehicle(frame, vehicle, vehicle_box)

            for plate in frame_plate_detections:
                draw_plate_detection(frame, plate.box, plate.text)

            counting_line.draw(frame)
            cv2.putText(
                frame,
                f"Vehicles crossed: {len(events)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Processing FPS: {output_fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            if args.show:
                cv2.imshow("Vehicle counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(
            csv_file,
            fieldnames=(
                "frame",
                "time_seconds",
                "track_id",
                "class_name",
                "center_x",
                "center_y",
                "plate_text",
            ),
        )
        writer_csv.writeheader()
        for event in events:
            writer_csv.writerow(
                {
                    "frame": event.frame,
                    "time_seconds": f"{event.time_seconds:.2f}",
                    "track_id": event.track_id,
                    "class_name": event.class_name,
                    "center_x": event.center_x,
                    "center_y": event.center_y,
                    "plate_text": event.plate_text,
                }
            )

    print(f"Processed video: {output_path}")
    print(f"CSV report: {csv_path}")
    print(f"Vehicles crossed: {len(events)}")
    print(f"Processed at: {output_fps:.1f} FPS")
    print(f"Vehicle classes: {args.classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
