"""
Mask generation script for blue cube using HSV thresholding.
Saves a binary mask for frame 0 (or any frame) to be used by FoundationPose.

Usage:
    python generate_mask.py --scene_dir demo_data/cubes
    python generate_mask.py --scene_dir demo_data/cubes --frame_idx 0 --interactive
"""

import os
import argparse
import numpy as np
import cv2


def get_blue_mask(color_bgr: np.ndarray, lower_hsv, upper_hsv) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def interactive_tune(color_bgr: np.ndarray):
    """
    Opens an OpenCV window with trackbars to tune HSV thresholds live.
    Press 's' to save and exit, 'q' to quit without saving.
    Returns (lower, upper) numpy arrays, or None if user quits.
    """
    window = 'HSV Tuner — press S to save, Q to quit'
    cv2.namedWindow(window)

    # Default starting values for blue
    cv2.createTrackbar('H min', window, 100, 179, lambda x: None)
    cv2.createTrackbar('H max', window, 130, 179, lambda x: None)
    cv2.createTrackbar('S min', window,  80, 255, lambda x: None)
    cv2.createTrackbar('S max', window, 255, 255, lambda x: None)
    cv2.createTrackbar('V min', window,  50, 255, lambda x: None)
    cv2.createTrackbar('V max', window, 255, 255, lambda x: None)

    while True:
        h_min = cv2.getTrackbarPos('H min', window)
        h_max = cv2.getTrackbarPos('H max', window)
        s_min = cv2.getTrackbarPos('S min', window)
        s_max = cv2.getTrackbarPos('S max', window)
        v_min = cv2.getTrackbarPos('V min', window)
        v_max = cv2.getTrackbarPos('V max', window)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = get_blue_mask(color_bgr, lower, upper)

        # Show original + mask side by side
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = color_bgr.copy()
        overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
        combined = np.hstack([overlay, mask_vis])
        # Resize if too large for screen
        h, w = combined.shape[:2]
        if w > 1800:
            scale = 1800 / w
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))
        cv2.imshow(window, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            cv2.destroyAllWindows()
            print(f'Saved thresholds — lower={lower.tolist()}, upper={upper.tolist()}')
            return lower, upper
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None


def click_hsv(color_bgr: np.ndarray):
    """
    Click on the image to print HSV values — handy for finding your threshold range.
    Press 'q' to close.
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv[y, x]
            print(f'  Clicked ({x}, {y}) → HSV = ({h}, {s}, {v})')

    cv2.namedWindow('Click to sample HSV — press Q to close')
    cv2.setMouseCallback('Click to sample HSV — press Q to close', on_mouse)
    cv2.imshow('Click to sample HSV — press Q to close', color_bgr)
    print('Click on the blue cube to read HSV values. Press Q when done.')
    while True:
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def load_frame(scene_dir: str, frame_idx: int) -> np.ndarray:
    color_dir = os.path.join(scene_dir, 'rgb')
    files = sorted([
        f for f in os.listdir(color_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not files:
        raise FileNotFoundError(f'No images found in {color_dir}')
    path = os.path.join(color_dir, files[frame_idx])
    print(f'Loading frame: {path}')
    img = cv2.imread(path)
    if img is None:
        raise IOError(f'Could not read image: {path}')
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--scene_dir', type=str, default=f'{code_dir}/demo_data/cubes')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Which frame to generate a mask for (default: 0)')
    parser.add_argument('--interactive', action='store_true',
                        help='Open HSV tuner GUI to dial in thresholds visually')
    parser.add_argument('--sample', action='store_true',
                        help='Click-to-sample mode: prints HSV values under cursor')
    parser.add_argument('--h_min', type=int, default=100)
    parser.add_argument('--h_max', type=int, default=130)
    parser.add_argument('--s_min', type=int, default=80)
    parser.add_argument('--s_max', type=int, default=255)
    parser.add_argument('--v_min', type=int, default=50)
    parser.add_argument('--v_max', type=int, default=255)
    args = parser.parse_args()

    color_bgr = load_frame(args.scene_dir, args.frame_idx)

    if args.sample:
        click_hsv(color_bgr)

    if args.interactive:
        lower, upper = interactive_tune(color_bgr)
        if lower is None:
            print('Tuning cancelled — no mask saved.')
            exit(0)
    else:
        lower = np.array([args.h_min, args.s_min, args.v_min])
        upper = np.array([args.h_max, args.s_max, args.v_max])

    mask = get_blue_mask(color_bgr, lower, upper)

    if not mask.any():
        print('WARNING: mask is empty! Try --interactive or --sample to tune thresholds.')
    else:
        coverage = mask.sum() / mask.size * 100
        print(f'Mask coverage: {coverage:.2f}% of frame pixels')

    out_dir = os.path.join(args.scene_dir, 'masks')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.frame_idx:04d}.png')
    cv2.imwrite(out_path, mask)
    print(f'Mask saved to: {out_path}')

    # Show final result
    overlay = color_bgr.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
    cv2.imshow('Final mask (any key to close)', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
