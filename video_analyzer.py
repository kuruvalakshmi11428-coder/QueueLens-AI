import cv2
import numpy as np


def analyze_video(video_path, sample_every=5):
    """Measure foreground motion using OpenCV background subtraction."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=True
    )

    scores = []
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            if frame_number % sample_every != 0:
                continue

            frame = cv2.resize(frame, (640, 360))
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            mask = subtractor.apply(blurred)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            moving_pixels = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            scores.append(100 * moving_pixels / total_pixels)
    finally:
        cap.release()

    if not scores:
        raise ValueError("No video frames could be analysed.")

    # Skip some initial frames while the background model adapts.
    stable_scores = scores[min(3, len(scores) - 1):]
    average_motion = float(np.mean(stable_scores))
    peak_motion = float(np.max(stable_scores))

    if average_motion < 2:
        activity = "Low"
    elif average_motion < 8:
        activity = "Moderate"
    else:
        activity = "High"

    return {
        "video_duration_seconds": round(duration, 2),
        "original_resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "analysed_frames": len(scores),
        "average_motion_percent": round(average_motion, 2),
        "peak_motion_percent": round(peak_motion, 2),
        "activity_level": activity,
        "limitation": (
            "Motion is an activity proxy, not a verified people count "
            "or direct measurement of queue length."
        )
    }
