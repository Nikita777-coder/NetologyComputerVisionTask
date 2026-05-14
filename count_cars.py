#!/usr/bin/env python3
"""Count cars crossing a line in a video with YOLOv8 and BoT-SORT."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


COCO_CAR_CLASS_ID = 2
DEFAULT_PLATE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "license_plate_detector.pt"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class TrackState:
    previous_center: Point | None = None
    counted: bool = False
    plate_text: str = ""


@dataclass
class CrossingEvent:
    frame: int
    time_seconds: float
    track_id: int
    class_name: str
    center_x: int
    center_y: int
    plate_text: str


class OptionalPlateRecognizer:
    """License plate detector/OCR that is used only when explicitly enabled."""

    def __init__(self, model_path: str | None, enabled: bool) -> None:
        self.enabled = enabled
        self.detector: YOLO | None = None
        self.reader = None

        if not enabled:
            return
        if not model_path:
            model_path = str(DEFAULT_PLATE_MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"License plate model was not found: {model_path}. "
                "Download it or pass another path with --plate-model."
            )

        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is required for plate recognition. Install it with "
                "`pip install -r requirements-plates.txt`."
            ) from exc

        self.detector = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=False)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in text.upper() if ch.isalnum())

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

    def recognize(self, car_crop: np.ndarray) -> str:
        if not self.enabled or self.detector is None or self.reader is None:
            return ""
        if car_crop.size == 0:
            return ""

        results = self.detector.predict(car_crop, verbose=False, conf=0.10)
        best_text = ""
        best_confidence = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_crop = car_crop[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
                if plate_crop.size == 0:
                    continue
                for variant in self._ocr_variants(plate_crop):
                    ocr_results = self.reader.readtext(
                        variant,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                    )
                    for _, text, confidence in ocr_results:
                        normalized = self._normalize_text(text)
                        if normalized and confidence > best_confidence:
                            best_text = normalized
                            best_confidence = float(confidence)

        return best_text


def parse_point(raw: str) -> Point:
    try:
        x_raw, y_raw = raw.split(",", maxsplit=1)
        return Point(float(x_raw), float(y_raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point must be formatted as x,y") from exc


def signed_side(point: Point, line_start: Point, line_end: Point) -> float:
    return (line_end.x - line_start.x) * (point.y - line_start.y) - (
        line_end.y - line_start.y
    ) * (point.x - line_start.x)


def movement_matches_direction(previous: Point, current: Point, direction: str) -> bool:
    dx = current.x - previous.x
    dy = current.y - previous.y

    if direction == "left-to-right":
        return dx > 0
    if direction == "right-to-left":
        return dx < 0
    if direction == "top-to-bottom":
        return dy > 0
    if direction == "bottom-to-top":
        return dy < 0
    raise ValueError(f"Unsupported direction: {direction}")


def crossed_line(
    previous: Point,
    current: Point,
    line_start: Point,
    line_end: Point,
    direction: str,
) -> bool:
    previous_side = signed_side(previous, line_start, line_end)
    current_side = signed_side(current, line_start, line_end)
    changed_side = previous_side == 0 or current_side == 0 or (previous_side * current_side < 0)
    return changed_side and movement_matches_direction(previous, current, direction)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, track, and count cars crossing a line in a video."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument("--output", default="vehicle_counter/output_counted.mp4")
    parser.add_argument("--csv", default="vehicle_counter/crossings.csv")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model name or path.")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence.")
    parser.add_argument("--line-start", type=parse_point, required=True, help="Line point x,y.")
    parser.add_argument("--line-end", type=parse_point, required=True, help="Line point x,y.")
    parser.add_argument(
        "--direction",
        choices=("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"),
        default="left-to-right",
        help="Only crossings moving in this direction are counted.",
    )
    parser.add_argument("--show", action="store_true", help="Show live preview window.")
    parser.add_argument("--enable-plates", action="store_true", help="Enable optional plate OCR.")
    parser.add_argument(
        "--plate-model",
        default=str(DEFAULT_PLATE_MODEL_PATH),
        help="YOLO model trained to detect license plates.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_path = Path(args.csv)

    if not input_path.exists():
        print(f"Input video does not exist: {input_path}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    plate_recognizer = OptionalPlateRecognizer(args.plate_model, args.enable_plates)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        print(f"Cannot open video: {input_path}", file=sys.stderr)
        return 2

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    track_states: dict[int, TrackState] = {}
    events: list[CrossingEvent] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=[COCO_CAR_CLASS_ID],
                conf=args.conf,
                verbose=False,
            )
            result = results[0] if results else None

            if result is not None and result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
                    center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                    state = track_states.setdefault(track_id, TrackState())

                    if (
                        state.previous_center is not None
                        and not state.counted
                        and crossed_line(
                            state.previous_center,
                            center,
                            args.line_start,
                            args.line_end,
                            args.direction,
                        )
                    ):
                        state.counted = True
                        car_crop = frame[y1:y2, x1:x2]
                        state.plate_text = plate_recognizer.recognize(car_crop)
                        events.append(
                            CrossingEvent(
                                frame=frame_index,
                                time_seconds=frame_index / fps,
                                track_id=track_id,
                                class_name=model.names.get(class_id, str(class_id)),
                                center_x=int(center.x),
                                center_y=int(center.y),
                                plate_text=state.plate_text,
                            )
                        )

                    state.previous_center = center
                    color = (0, 180, 0) if state.counted else (255, 120, 0)
                    label = f"ID {track_id} {model.names.get(class_id, class_id)}"
                    if state.plate_text:
                        label += f" {state.plate_text}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (int(center.x), int(center.y)), 4, (0, 255, 255), -1)
                    draw_label(frame, label, x1, max(20, y1), color)

            cv2.line(
                frame,
                (int(args.line_start.x), int(args.line_start.y)),
                (int(args.line_end.x), int(args.line_end.y)),
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Cars crossed: {len(events)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
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
    print(f"Cars crossed: {len(events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
