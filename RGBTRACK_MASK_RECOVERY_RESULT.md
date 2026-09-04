# RGBTrack Mask-Guided Recovery Result

Date: 2026-09-04

## Success criteria

The target pipeline is:

```text
first-frame registration
-> XMem mask propagation
-> deterministic CAD silhouette
-> mask/pose consistency detection
-> automatic re-registration after drift
-> continue tracking
```

The final validation must complete all 300 frames of `demo_data/zhuying`, keep the XMem mask valid, detect the known RGB-only drift, recover from it with the current XMem mask, and finish without continuing along the wrong pose.

## Final recommended configuration

- Mode: `0` (RGB-only tracking)
- Shorter side: `640`
- Tracking refinement iterations: `3`
- XMem: enabled
- Center-distance threshold: `0.08` of image diagonal
- Mask/CAD IoU threshold: `0.30`
- Relocalization cooldown: `5` frames
- Debug: `2`

Recommended command:

```bash
conda activate rgbtrack
cd /home/wlsea1/j_ws/src/GetPose/src/RGBTrack

python run_demo_without_depth.py \
  --mesh_file demo_data/zhuying/mesh/mid_haican_m_simplified.obj \
  --test_scene_dir demo_data/zhuying \
  --mode 0 \
  --shorter_side 640 \
  --track_refine_iter 3 \
  --use_xmem \
  --relocalize_center_ratio 0.08 \
  --relocalize_iou 0.30 \
  --relocalize_cooldown 5 \
  --debug 2 \
  --debug_dir debug/final/m0_s640_i3_xmem
```

## XMem weights

The local weight file is:

```text
XMem/saves/XMem.pth
```

Validation:

- File size: `249026057` bytes
- MD5: `a0c894e24ca02440a3b7b8b4fa6983dc`
- The size and MD5 match the official GitHub release asset.
- `torch.load(..., map_location="cpu")` succeeded.
- `XMem(...)` initialization succeeded.
- `XMem/.gitignore` ignores `saves/`, so the weight file is not added to Git.

## XMem propagation result

The 300-frame XMem-only validation completed successfully.

Final validation output counts:

- XMem masks: `300/300`
- XMem overlays: `300/300`
- Pose visualizations: `300/300`
- Saved poses: `300/300`
- Valid XMem mask frames: `300/300`

Selected propagated mask areas:

| Frame | Mask area (px) |
| ---: | ---: |
| 0 | 1450 |
| 50 | 1257 |
| 100 | 931 |
| 150 | 1731 |
| 200 | 5549 |
| 250 | 14124 |
| 299 | 23393 |

The mask remains non-empty and spatially continuous through the sequence. Its increasing area in the latter half follows the object becoming substantially larger in the image. No empty-mask or resolution-shape crash occurred.

## Deterministic CAD silhouette

The previous `render_cad_mask()` used random mesh vertices plus `cv2.fillPoly`, which is unsuitable for a recovery criterion.

The recovery pipeline now renders the silhouette with project-native nvdiffrast:

```text
nvdiffrast_render_depthonly(...)
-> depth > 0
-> boolean CAD silhouette
```

Important implementation details:

- Uses `reader.K`.
- Uses `reader.W` and `reader.H`.
- Does not hard-code 640 x 480.
- Uses the original CAD mesh coordinate system corresponding to the pose returned by `FoundationPose.register()/track_one()`.
- Reuses a CUDA rasterization context and prebuilt original-mesh tensors.
- Repeated rendering of the same pose produces an identical mask.

A coordinate-system issue was found during stage 4: `est.mesh_tensors` contains FoundationPose's internally centered mesh, while the pose returned to the demo corresponds to the original CAD coordinates. Pairing those two caused frame-0 IoU to be zero. Using tensors generated from the original mesh fixed frame 0 to approximately:

- IoU: `0.871`
- Center-distance ratio: `0.0010`

## Consistency metrics

Each frame records:

- mask center
- CAD silhouette center
- center distance in pixels
- center distance normalized by image diagonal
- mask/CAD IoU
- mask area
- CAD mask area
- mask validity
- tracking state
- pre-relocalization IoU/center ratio
- cumulative relocalization counters

The CSV is:

```text
debug/final/m0_s640_i3_xmem/tracking_metrics.csv
```

The final CSV has exactly 300 unique sequential frame IDs from 0 through 299.

## Drift detection

The no-recovery consistency-only run showed a clear transition:

| Frame | IoU | Center ratio | State |
| ---: | ---: | ---: | --- |
| 18 | 0.872 | 0.0004 | TRACK |
| 19 | 0.846 | 0.0014 | TRACK |
| 20 | 0.073 | 0.0364 | LOST |
| 21 | 0.023 | 0.048 | LOST |
| 25 | 0.000 | 0.057 | LOST |
| 50 | 0.000 | 0.097 | LOST |

Therefore the default thresholds were kept:

- `center_distance_ratio > 0.08` -> bad
- `mask_iou < 0.30` -> bad

The IoU threshold detects this failure earlier than the center threshold. The thresholds were not altered to force a desired result.

## Automatic relocalization result

The final 300-frame run produced:

- First drift/recovery event: frame `20`
- Relocalization attempts: `1`
- Successful relocalizations: `1`
- Failed relocalizations: `0`
- Invalid-mask frames: `0`
- Losses after successful recovery: `0`

Frame 20 before re-registration:

- IoU: `0.07296`
- Center-distance ratio: `0.03635`

Frame 20 after re-registration with the current XMem mask:

- IoU: `0.61712`
- Center-distance ratio: `0.00815`
- State: `RELOCALIZE_OK`

The following frames returned to `TRACK`. No second relocalization was required.

Final frame 299:

- IoU: `0.91118`
- Center-distance ratio: `0.00395`
- State: `TRACK`

This demonstrates the intended behavior:

```text
pose drifts away from XMem mask
-> consistency check detects failure
-> current RGB + current XMem mask are passed to binary_search_depth(...)
-> FoundationPose register() resets its internal tracking state
-> CAD pose returns to the object
-> subsequent track_one() continues from the recovered pose
```

## Final metric distribution

Post-recovery metrics across all 300 final-run frames:

- IoU minimum: `0.4949`
- IoU median: `0.8461`
- IoU mean: `0.7836`
- IoU maximum: `0.9391`
- Center-ratio minimum: `0.00011`
- Center-ratio median: `0.00183`
- Center-ratio mean: `0.00300`
- Center-ratio maximum: `0.01096`

All post-recovery metrics remain comfortably inside the configured LOST thresholds.

## Throughput

Approximate throughput is estimated from the timestamp span between the first and last saved pose in the final debug directory:

- Final XMem + consistency + automatic recovery pipeline: approximately `4.60 FPS`
- Existing A1 baseline, mode 0 / 480 / iter 1: approximately `13.00 FPS`
- Existing B2 baseline, mode 0 / 640 / iter 3: approximately `9.06 FPS`

The final number includes XMem inference, deterministic CAD silhouette rendering, consistency calculation, one re-registration event, and debug level 2 file output. It is therefore not directly equivalent to the earlier tracking-only baseline timing, but it shows the current accuracy/recovery cost.

## Difference from the previous baseline

Existing baseline B2 (`mode 0 / 640 / iter 3`) completed 300 frames but continued accumulating pose error; its projected CAD center first left the image at frame 298.

With XMem-guided recovery:

- the disagreement is detected at frame 20 instead of being allowed to accumulate;
- the current XMem mask is used to re-register once;
- the re-registered pose passes the same consistency test;
- tracking then remains aligned through frame 299;
- final IoU is approximately 0.911 instead of the CAD pose continuing to drift away.

No additional mode-1 baseline run was needed for this stage.

## Stage commits

- `1f85d62` — `fix: resolve XMem path from source directory`
- `657b420` — `feat: propagate object mask with XMem`
- `429dcba` — `feat: evaluate pose consistency with tracked mask`
- `e528709` — `feat: relocalize pose when tracking drifts`

The XMem weight validation stage does not have a Git commit because the official `XMem.pth` is intentionally ignored and must not be versioned.

## Modified files

Tracked project files changed by the mask-recovery work:

- `xmem_wrapper.py`
- `tools.py`
- `run_demo_without_depth.py`
- `RGBTRACK_MASK_RECOVERY_RESULT.md`

Local ignored runtime asset:

- `XMem/saves/XMem.pth`

## Conclusion

The requested RGB-only recovery loop is now working on the 300-frame sea-cucumber sequence:

```text
first-frame registration
-> XMem mask continuous propagation
-> deterministic CAD silhouette
-> IoU / center consistency check
-> drift detection
-> automatic re-registration from the current XMem mask
-> continued tracking
```

For the validated sequence, the first drift occurs at frame 20, one automatic re-registration succeeds, no further loss occurs, and frame 299 remains aligned with the tracked sea-cucumber mask.
