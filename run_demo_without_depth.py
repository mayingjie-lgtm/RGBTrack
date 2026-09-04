# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import csv
import time
from estimater import *
from datareader import *
import argparse
from tools import *
from xmem_wrapper import XMemMaskTracker, overlay_davis
import numpy as np

SAVE_VIDEO=False


def assert_frame_shapes(color, expected_hw, mask=None, depth=None):
    """Validate RGB, mask, and pseudo-depth against the reader output size."""
    assert color.shape[:2] == expected_hw, (
        f"RGB shape {color.shape[:2]} does not match reader shape {expected_hw}"
    )
    if mask is not None:
        assert mask.shape[:2] == expected_hw, (
            f"Mask shape {mask.shape[:2]} does not match reader shape {expected_hw}"
        )
    if depth is not None:
        assert depth.shape[:2] == expected_hw, (
            f"Depth shape {depth.shape[:2]} does not match reader shape {expected_hw}"
        )


def normalize_xmem_mask(mask, expected_hw):
    """Return a boolean XMem mask at reader resolution and flag tiny masks invalid."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != expected_hw:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (expected_hw[1], expected_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    min_pixels = max(16, int(expected_hw[0] * expected_hw[1] * 1e-4))
    return mask, int(mask.sum()) >= min_pixels


def save_xmem_debug(debug_dir, frame_name, color, mask):
    """Save the propagated binary mask and an RGB overlay for visual inspection."""
    mask_dir = os.path.join(debug_dir, "xmem_mask")
    vis_dir = os.path.join(debug_dir, "xmem_vis")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    imageio.imwrite(os.path.join(mask_dir, f"{frame_name}.png"), mask.astype(np.uint8) * 255)
    imageio.imwrite(
        os.path.join(vis_dir, f"{frame_name}.png"),
        overlay_davis(color, mask.astype(np.uint8)),
    )


def get_mask_center(mask):
    """Return the pixel centroid of a binary mask, or NaNs for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def calculate_tracking_metrics(current_mask, cad_mask, mask_valid, image_width, image_height):
    """Measure geometric agreement between the tracked mask and rendered CAD silhouette."""
    mask_area = int(np.sum(current_mask)) if current_mask is not None else 0
    cad_mask_area = int(np.sum(cad_mask))
    mask_center_x, mask_center_y = (
        get_mask_center(current_mask) if current_mask is not None else (float("nan"), float("nan"))
    )
    cad_center_x, cad_center_y = get_mask_center(cad_mask)

    centers_valid = mask_valid and mask_area > 0 and cad_mask_area > 0
    if centers_valid:
        center_distance_px = float(
            np.hypot(mask_center_x - cad_center_x, mask_center_y - cad_center_y)
        )
        center_distance_ratio = center_distance_px / float(
            np.hypot(image_width, image_height)
        )
        mask_iou = float(compute_iou(current_mask, cad_mask))
    else:
        center_distance_px = float("inf")
        center_distance_ratio = float("inf")
        mask_iou = 0.0

    return {
        "mask_center_x": mask_center_x,
        "mask_center_y": mask_center_y,
        "cad_center_x": cad_center_x,
        "cad_center_y": cad_center_y,
        "center_distance_px": center_distance_px,
        "center_distance_ratio": center_distance_ratio,
        "mask_iou": mask_iou,
        "mask_area": mask_area,
        "cad_mask_area": cad_mask_area,
        "mask_valid": bool(mask_valid),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument(
        "--use_xmem",
        action="store_true",
        help="Propagate the first-frame object mask with XMem for diagnostics.",
    )
    parser.add_argument(
        "--shorter_side",
        type=int,
        default=480,
        help="Resize the shorter image side to limit pose-scoring GPU memory use.",
    )
    parser.add_argument(
        "--relocalize_center_ratio",
        type=float,
        default=0.08,
        help="Mark pose tracking lost above this normalized mask-center error.",
    )
    parser.add_argument(
        "--relocalize_iou",
        type=float,
        default=0.30,
        help="Mark pose tracking lost below this mask/CAD silhouette IoU.",
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    show_window = debug >= 1 and bool(os.environ.get("DISPLAY"))
    if debug >= 1 and not show_window:
        logging.info("No DISPLAY detected; saving debug visualization without cv2.imshow")
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    cad_mesh_tensors = make_mesh_tensors(mesh)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=args.shorter_side, zfar=np.inf
    )
    xmem_tracker = XMemMaskTracker() if args.use_xmem else None
    metrics_path = os.path.join(debug_dir, "tracking_metrics.csv")
    metrics_file = open(metrics_path, "w", newline="")
    metrics_fields = [
        "frame_id",
        "mask_center_x",
        "mask_center_y",
        "cad_center_x",
        "cad_center_y",
        "center_distance_px",
        "center_distance_ratio",
        "mask_iou",
        "mask_area",
        "cad_mask_area",
        "mask_valid",
        "tracking_state",
    ]
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
    metrics_writer.writeheader()

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        expected_hw = (reader.H, reader.W)
        assert_frame_shapes(color, expected_hw)
        current_mask = None
        mask_valid = False
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            assert_frame_shapes(color, expected_hw, mask=mask)
            last_mask= mask
            current_mask, mask_valid = normalize_xmem_mask(mask, expected_hw)
            if xmem_tracker is not None:
                xmem_tracker.initialize(color, mask)
                logging.info(
                    f"[XMEM] frame={i} valid={mask_valid} area={int(current_mask.sum())}"
                )
                if debug >= 2:
                    save_xmem_debug(debug_dir, reader.id_strs[i], color, current_mask)
            t1=time.time()
            pose= binary_search_depth(
                est,
                mesh,
                color,
                mask,
                reader.K,
                w=reader.W,
                h=reader.H,
                debug=True,
            )

            # pose = est.register_without_depth(
            #     K=reader.K,
            #     rgb=color,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            logging.info(f"Initial pose:\n{pose}")
        
            t2=time.time()
            if SAVE_VIDEO:
                output_video_path = "fp_nodepth_improved.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))
        else:
            if xmem_tracker is not None:
                current_mask = xmem_tracker.propagate(color)
                current_mask, mask_valid = normalize_xmem_mask(current_mask, expected_hw)
                logging.info(
                    f"[XMEM] frame={i} valid={mask_valid} area={int(current_mask.sum())}"
                )
                if debug >= 2:
                    save_xmem_debug(debug_dir, reader.id_strs[i], color, current_mask)
            t1=time.time()
            if args.mode==0:
                last_depth = np.zeros_like(last_mask)
            elif args.mode==1:
                last_depth = render_cad_depth(
                    pose,
                    mesh,
                    reader.K,
                    w=reader.W,
                    h=reader.H,
                )
            assert_frame_shapes(color, expected_hw, depth=last_depth)
            pose = est.track_one(
                rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter
            )
            t2=time.time()

        cad_mask = render_cad_silhouette(
            pose,
            mesh,
            reader.K,
            w=reader.W,
            h=reader.H,
            glctx=glctx,
            mesh_tensors=cad_mesh_tensors,
        )
        assert_frame_shapes(color, expected_hw, mask=cad_mask)
        metrics = calculate_tracking_metrics(
            current_mask,
            cad_mask,
            mask_valid,
            image_width=reader.W,
            image_height=reader.H,
        )
        pose_bad = (
            metrics["center_distance_ratio"] > args.relocalize_center_ratio
            or metrics["mask_iou"] < args.relocalize_iou
        )
        if not mask_valid:
            tracking_state = "LOST_MASK"
        elif pose_bad:
            tracking_state = "LOST"
        else:
            tracking_state = "TRACK"
        metrics_writer.writerow(
            {"frame_id": i, **metrics, "tracking_state": tracking_state}
        )
        metrics_file.flush()

        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color=cv2.putText(color, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            color=cv2.putText(
                color,
                f"{tracking_state} IoU {metrics['mask_iou']:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            color=cv2.putText(
                color,
                f"center err {metrics['center_distance_px']:.1f}px ({metrics['center_distance_ratio']:.3f})",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            vis = draw_posed_3d_box(
                reader.K, img=color, ob_in_cam=center_pose, bbox=bbox
            )
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            if show_window:
                cv2.imshow("1", vis[..., ::-1])
                cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
        if SAVE_VIDEO:
            video_writer.write(vis[..., ::-1])

    metrics_file.close()
