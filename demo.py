"""
================================================================
  Moneyball Football
  From a Movie to a Model — Player Tracking Demo

  Author  : Emrehan
  Stack   : Python 3.12 | YOLOv11 | OpenCV | ByteTrack
  Purpose : Architecture demo for LinkedIn
            Full implementation is not included.

  Pipeline:
    VideoProcessor -> ObjectDetector -> PlayerTracker -> Visualizer
                                                      -> [StatisticsModule *]

  * Phase 2 — coming soon:
      Instantaneous speed (km/h)
      Total distance covered (m)
      Sprint detection
      Homography-based pixel-to-meter calibration
================================================================
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO
from pathlib import Path


# ---------------------------------------------------------------
# Config  —  single place to tune the pipeline
# ---------------------------------------------------------------

class Config:
    MODEL        = "yolo11n.pt"       # yolo11s.pt for better accuracy
    CONFIDENCE   = 0.35
    TRACKER      = "bytetrack.yaml"
    DEVICE       = ""                 # "" = auto (CUDA > MPS > CPU)
    TRAIL_LEN    = 30                 # frames of motion history
    TARGET_CLS   = {"person", "sports ball"}
    INPUT_VIDEO  = "input.mp4"
    OUTPUT_VIDEO = "output.mp4"


# ---------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------

@dataclass
class Detection:
    track_id  : int
    cls_name  : str
    bbox      : tuple[float, float, float, float]
    confidence: float
    center    : tuple[float, float]


@dataclass
class PlayerState:
    track_id  : int
    trajectory: deque = field(default_factory=lambda: deque(maxlen=Config.TRAIL_LEN))

    # -- Phase 2 placeholders ----------------------------------
    # speed_kmh     : float = 0.0
    # distance_m    : float = 0.0
    # sprint_count  : int   = 0
    # ----------------------------------------------------------


# ---------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------

class VideoProcessor:
    """Handles video I/O. Decoupled from all business logic."""

    def __init__(self, input_path: str, output_path: str):
        self.input_path  = Path(input_path)
        self.output_path = Path(output_path)
        self.cap         = None
        self.writer      = None
        self.fps         = 30.0
        self.width       = 0
        self.height      = 0
        self.total_frames= 0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.input_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        print(f"[Video] {self.width}x{self.height} @ {self.fps:.0f}fps — {self.total_frames} frames")

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def write_frame(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        if self.cap:    self.cap.release()
        if self.writer: self.writer.release()


# ---------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------

class ObjectDetector:
    """
    Wraps YOLO tracking. Filters by target classes only.
    Swap model_name to upgrade detection quality with zero other changes.
    """

    def __init__(self):
        self.model = YOLO(Config.MODEL)
        self.target_ids = [
            i for i, n in self.model.names.items()
            if n in Config.TARGET_CLS
        ]
        print(f"[Detector] {Config.MODEL} | tracking: {list(Config.TARGET_CLS)}")

    def run(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist   = True,
            conf      = Config.CONFIDENCE,
            classes   = self.target_ids,
            tracker   = Config.TRACKER,
            device    = Config.DEVICE,
            verbose   = False,
        )

        detections = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        for box, tid, cid, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            detections.append(Detection(
                track_id   = int(tid),
                cls_name   = self.model.names[int(cid)],
                bbox       = (x1, y1, x2, y2),
                confidence = float(conf),
                center     = ((x1 + x2) / 2, (y1 + y2) / 2),
            ))

        return detections


# ---------------------------------------------------------------
# PlayerTracker
# ---------------------------------------------------------------

class PlayerTracker:
    """
    Maintains per-ID state across frames.
    Acts as the data provider for the Statistics module (Phase 2).
    """

    def __init__(self):
        self.states: dict[int, PlayerState] = {}

    def update(self, detections: list[Detection]) -> None:
        for det in detections:
            if det.track_id not in self.states:
                self.states[det.track_id] = PlayerState(track_id=det.track_id)
            self.states[det.track_id].trajectory.append(det.center)

            # -- Phase 2 hook (not implemented in demo) ----------
            # stats_module.update(self.states[det.track_id], det)
            # ----------------------------------------------------

    def get(self, track_id: int) -> Optional[PlayerState]:
        return self.states.get(track_id)


# ---------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------

class Visualizer:
    """Pure rendering — no business logic, no state."""

    COLORS = {
        "person":      (220, 180, 60),
        "sports ball": (60, 220, 255),
    }
    TRAIL_START = (160, 160, 160)
    TRAIL_END   = (0, 200, 255)

    def render(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracker: PlayerTracker,
    ) -> np.ndarray:
        canvas = frame.copy()

        for det in detections:
            state = tracker.get(det.track_id)
            if state and len(state.trajectory) > 1:
                self._trail(canvas, state)

        for det in detections:
            self._box(canvas, det)

        return canvas

    def _trail(self, frame: np.ndarray, state: PlayerState) -> None:
        pts = list(state.trajectory)
        n   = len(pts)
        for i in range(1, n):
            t     = i / n
            color = tuple(int(self.TRAIL_START[c] * (1-t) + self.TRAIL_END[c] * t) for c in range(3))
            cv2.line(
                frame,
                (int(pts[i-1][0]), int(pts[i-1][1])),
                (int(pts[i][0]),   int(pts[i][1])),
                color, max(1, int(t * 3)), cv2.LINE_AA
            )
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 4, self.TRAIL_END, -1, cv2.LINE_AA)

    def _box(self, frame: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = map(int, det.bbox)
        color  = self.COLORS.get(det.cls_name, (200, 200, 200))
        clen   = max(10, int((x2 - x1) * 0.15))

        for (px, py), (dx, dy) in zip(
            [(x1,y1),(x2,y1),(x1,y2),(x2,y2)],
            [(1,1),(-1,1),(1,-1),(-1,-1)]
        ):
            cv2.line(frame, (px, py), (px + dx*clen, py),   color, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy*clen),   color, 2, cv2.LINE_AA)

        label = f"#{det.track_id} {det.cls_name}"
        # Phase 2: label = f"#{det.track_id} | {speed:.1f} km/h | {dist:.0f}m"

        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), bl = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(y1 - 6, th + 6)

        cv2.rectangle(frame, (x1, ty - th - bl - 4), (x1 + tw + 6, ty), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty - bl - 2), font, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


# ---------------------------------------------------------------
# StatisticsModule  [Phase 2 — stub]
# ---------------------------------------------------------------

class StatisticsModule:
    """
    Phase 2: Pixel-to-meter conversion via Homography + per-player stats.

    How it will work:
        1. User marks 4 pitch line intersections (known real-world coords).
        2. cv2.findHomography() builds the perspective transform matrix.
        3. Every player center is projected to real-world meters.
        4. Speed and distance are derived from consecutive positions.

    This class is a stub — full implementation in Phase 2.
    """

    def __init__(self, homography_matrix: Optional[np.ndarray] = None):
        self.H = homography_matrix    # None until calibrated

    def pixel_to_meter(self, point: tuple[float, float]) -> tuple[float, float]:
        """Project pixel (cx, cy) to real-world (x_m, y_m)."""
        if self.H is None:
            return point              # passthrough until calibrated
        # TODO: cv2.perspectiveTransform implementation
        raise NotImplementedError("Phase 2")

    def compute_speed(self, state: PlayerState, fps: float) -> float:
        """Returns instantaneous speed in km/h."""
        # TODO: Euclidean(p[-1], p[-2]) -> meters -> * fps * 3.6
        raise NotImplementedError("Phase 2")

    def compute_distance(self, state: PlayerState) -> float:
        """Returns cumulative distance in meters."""
        # TODO: sum of consecutive Euclidean distances in metric space
        raise NotImplementedError("Phase 2")


# ---------------------------------------------------------------
# FootballAnalytics  (Orchestrator)
# ---------------------------------------------------------------

class FootballAnalytics:
    """
    Wires all components and runs the frame loop.

    Extension points:
        StatisticsModule  ->  speed, distance, sprint detection
        TeamClassifier    ->  color-based team separation
        HomographyMapper  ->  perspective calibration
        EventDetector     ->  pass, shot, ball-loss events
    """

    def __init__(self):
        self.video    = VideoProcessor(Config.INPUT_VIDEO, Config.OUTPUT_VIDEO)
        self.detector = ObjectDetector()
        self.tracker  = PlayerTracker()
        self.viz      = Visualizer()
        self.stats    = StatisticsModule()    # stub — Phase 2

    def _overlay(self, frame: np.ndarray, idx: int, dets: list[Detection]) -> np.ndarray:
        lines = [
            f"Frame: {idx}",
            f"Players: {sum(1 for d in dets if d.cls_name == 'person')}",
            f"Ball: {sum(1 for d in dets if d.cls_name == 'sports ball')}",
            # Phase 2: f"Avg Speed: {avg_speed:.1f} km/h"
        ]
        bg = frame.copy()
        cv2.rectangle(bg, (8, 8), (165, 8 + len(lines) * 22 + 10), (20, 20, 20), -1)
        cv2.addWeighted(bg, 0.55, frame, 0.45, 0, dst=frame)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (14, 28 + i*22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        return frame

    def run(self) -> None:
        self.video.open()
        print("[Pipeline] Starting...")

        try:
            idx = 0
            while True:
                ret, frame = self.video.read_frame()
                if not ret:
                    break

                detections   = self.detector.run(frame)
                self.tracker.update(detections)
                canvas       = self.viz.render(frame, detections, self.tracker)
                canvas       = self._overlay(canvas, idx, detections)
                self.video.write_frame(canvas)

                if idx % 60 == 0:
                    pct = idx / self.video.total_frames * 100 if self.video.total_frames else 0
                    print(f"  {idx:>5} frames | {pct:.1f}% | tracked: {len(self.tracker.states)}")
                idx += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.video.close()
            print(f"Done. Saved -> {Config.OUTPUT_VIDEO}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    FootballAnalytics().run()
