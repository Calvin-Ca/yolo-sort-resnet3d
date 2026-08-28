#!/usr/bin/env python3
"""Zero-copy DeepStream YOLO, NvSORT, and tracked 3D action pipeline."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

import pyds

from run_yolo_tracker import (
    MetadataWriter,
    build_pipeline,
    render_infer_config,
    video_size,
)


ACTION_NAMES = {
    0: "clean",
    1: "concrete",
    2: "formwork",
    3: "prepare",
    4: "rebar",
    5: "rest/talk",
    6: "scaffold",
    7: "transport",
    8: "walk",
}
UNTRACKED_OBJECT_ID = (1 << 64) - 1


def video_fps(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    numerator, denominator = (
        subprocess.check_output(command, text=True).strip().split("/")
    )
    return float(numerator) / float(denominator)


def render_template(template, destination, replacements):
    text = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(key, str(value))
    destination.write_text(text, encoding="utf-8")


class TrackedRoiExpander:
    """Expand tracked boxes for preprocessing, then restore display metadata."""

    def __init__(self, frame_width, frame_height, scale=2.0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = scale
        self.original = {}
        self.restore_misses = 0
        self.debug_count = 0

    def _for_each_object(self, info, callback):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            object_list = frame_meta.obj_meta_list
            while object_list is not None:
                try:
                    object_meta = pyds.NvDsObjectMeta.cast(object_list.data)
                except StopIteration:
                    break
                callback(frame_meta, object_meta)
                try:
                    object_list = object_list.next
                except StopIteration:
                    break
            try:
                frame_list = frame_list.next
            except StopIteration:
                break

    def clear_pending(self):
        """Drop metadata snapshots that were not returned by an async branch."""
        self.original.clear()

    def expand(self, _pad, info):
        def expand_object(frame_meta, object_meta):
            track_id = int(object_meta.object_id)
            if track_id == UNTRACKED_OBJECT_ID:
                return
            if os.environ.get("CAIC_DEBUG_TRACKS") and self.debug_count < 20:
                print(
                    f"TRACK_DEBUG frame={int(frame_meta.frame_num)} id={track_id} "
                    f"roi={object_meta.rect_params.left:.1f},"
                    f"{object_meta.rect_params.top:.1f},"
                    f"{object_meta.rect_params.width:.1f},"
                    f"{object_meta.rect_params.height:.1f}",
                    file=sys.stderr,
                    flush=True,
                )
                self.debug_count += 1
            rect = object_meta.rect_params
            key = (int(frame_meta.frame_num), track_id)
            self.original[key] = (
                float(rect.left),
                float(rect.top),
                float(rect.width),
                float(rect.height),
            )
            center_x = float(rect.left) + float(rect.width) / 2
            center_y = float(rect.top) + float(rect.height) / 2
            side = min(
                max(float(rect.width), float(rect.height)) * self.scale,
                self.frame_width,
                self.frame_height,
            )
            left = min(
                max(0.0, center_x - side / 2), self.frame_width - side
            )
            top = min(
                max(0.0, center_y - side / 2), self.frame_height - side
            )
            rect.left = left
            rect.top = top
            rect.width = side
            rect.height = side

        self._for_each_object(info, expand_object)
        return Gst.PadProbeReturn.OK

    def restore(self, _pad, info):
        def restore_object(frame_meta, object_meta):
            key = (int(frame_meta.frame_num), int(object_meta.object_id))
            original = self.original.pop(key, None)
            if original is None:
                self.restore_misses += 1
                return
            rect = object_meta.rect_params
            rect.left, rect.top, rect.width, rect.height = original

        self._for_each_object(info, restore_object)
        return Gst.PadProbeReturn.OK


def parse_args():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Native DeepStream YOLO + NvSORT + 3D actions"
    )
    parser.add_argument(
        "--input", type=Path, default=root / "benchmark/e2e/clip1_30s.mp4"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmark/deepstream/yolo_tracker_action_native.mp4",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=root / "yolo/edge_pt/onnx/worker.640.yolov5n6.fp16.engine",
    )
    parser.add_argument(
        "--action-engine",
        type=Path,
        default=root / "workflow/resource/action_resnet3d_t30_s200.fp16.engine",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--action-stride", type=int, default=30)
    parser.add_argument("--roi-scale", type=float, default=2.0)
    parser.add_argument(
        "--action-subsample", type=int, default=0, choices=(0, 1, 2),
        help="Use every N+1th frame for the 3D action cache",
    )
    parser.add_argument(
        "--action-async", action="store_true",
        help="Enable asynchronous secondary action inference",
    )
    parser.add_argument("--bitrate", type=int, default=10_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.engine = args.engine.resolve()
    args.action_engine = args.action_engine.resolve()
    args.metadata = (
        args.metadata.resolve()
        if args.metadata
        else args.output.with_suffix(".tracks.jsonl")
    )
    args.actions = (
        args.actions.resolve()
        if args.actions
        else args.output.with_suffix(".actions.jsonl")
    )
    args.summary = (
        args.summary.resolve()
        if args.summary
        else args.output.with_suffix(".summary.json")
    )
    detector_config = args.summary.with_suffix(".detector.txt")
    args.action_preprocess_config = args.summary.with_suffix(".preprocess.txt")
    args.action_infer_config = args.summary.with_suffix(".action-infer.txt")

    required = [
        args.input,
        args.engine,
        args.action_engine,
        script_dir / "libnvdsinfer_custom_yolov5.so",
        script_dir / "libnvds_custom_sequence_track_id.so",
        script_dir / "labels.txt",
        script_dir / "action_labels.txt",
        script_dir / "config_infer_yolov5.txt.in",
        script_dir / "config_preprocess_action_track.txt.in",
        script_dir / "config_infer_action_track.txt.in",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    generated = [
        args.output,
        args.metadata,
        args.actions,
        args.summary,
        detector_config,
        args.action_preprocess_config,
        args.action_infer_config,
    ]
    for path in generated:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    input_width, input_height = video_size(args.input)
    fps = video_fps(args.input)
    mux_width = (input_width + 7) // 8 * 8
    mux_height = (input_height + 1) // 2 * 2
    render_infer_config(
        script_dir / "config_infer_yolov5.txt.in",
        detector_config,
        args.engine,
        script_dir / "labels.txt",
        script_dir / "libnvdsinfer_custom_yolov5.so",
    )
    render_template(
        script_dir / "config_preprocess_action_track.txt.in",
        args.action_preprocess_config,
        {
            "@SEQUENCE_LIBRARY@": (
                script_dir / "libnvds_custom_sequence_track_id.so"
            ),
            "@ACTION_SUBSAMPLE@": args.action_subsample,
        },
    )
    render_template(
        script_dir / "config_infer_action_track.txt.in",
        args.action_infer_config,
        {
            "@ACTION_ENGINE@": args.action_engine,
            "@ACTION_LABELS@": script_dir / "action_labels.txt",
            "@ACTION_ASYNC@": 1 if args.action_async else 0,
        },
    )

    Gst.init(None)
    args.roi_expander = TrackedRoiExpander(
        mux_width, mux_height, scale=args.roi_scale
    )
    writer = MetadataWriter(
        args.metadata,
        action_jsonl_path=args.actions,
        action_names=ACTION_NAMES,
        action_stride=args.action_stride,
        fps=fps,
    )
    pipeline = build_pipeline(
        args, detector_config, writer, mux_width, mux_height
    )
    loop = GLib.MainLoop()
    error_message = None
    eos_seen = False
    drain_started = None

    def on_bus_message(_bus, message):
        nonlocal error_message
        nonlocal eos_seen, drain_started
        if message.type == Gst.MessageType.EOS:
            if not eos_seen:
                eos_seen = True
                drain_started = time.perf_counter()
                # Allow asynchronous secondary inference callbacks to finish.
                GLib.timeout_add(1000, finish_after_drain)
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            error_message = f"{error}: {debug}"
            loop.quit()
        return True

    def finish_after_drain():
        args.roi_expander.clear_pending()
        loop.quit()
        return False

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)
    started = time.perf_counter()
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Unable to set the pipeline to PLAYING")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        writer.close()
    elapsed = time.perf_counter() - started
    if error_message:
        raise RuntimeError(error_message)

    summary = writer.summary(elapsed, args.input, args.output)
    summary.update(
        {
            "input_width": input_width,
            "input_height": input_height,
            "output_width": mux_width,
            "output_height": mux_height,
            "metadata": str(args.metadata),
            "actions": str(args.actions),
            "action_engine": str(args.action_engine),
            "action_profile": {
                "window_length": 30,
                "input_size": 200,
                "stride": args.action_stride,
                "subsample": args.action_subsample,
                "roi_scale": args.roi_scale,
            },
            "roi_restore_misses": args.roi_expander.restore_misses,
            "roi_restore_pending": len(args.roi_expander.original),
            "action_async": args.action_async,
            "async_drain_seconds": round(
                max(0.0, time.perf_counter() - drain_started)
                if drain_started else 0.0, 6
            ),
        }
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("NATIVE_ACTION_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
