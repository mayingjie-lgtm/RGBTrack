# RGBTrack A/B Baseline Result

Date: 2026-09-04

## Success criteria

- Run all six requested configurations on the same 300-frame `demo_data/zhuying` sequence.
- Keep each run in an independent `debug/ab/<case>` directory.
- Verify completion and capture an approximate tracking-only FPS.
- Record conservative objective drift evidence without changing the tracking algorithm.

## Dataset

- Mesh: `demo_data/zhuying/mesh/mid_haican_m_simplified.obj`
- Scene: `demo_data/zhuying`
- Frames: 300
- Input source: 1920 x 1536
- Debug visualization: `--debug 2`, saved for every frame.

## Results

| Case | Mode | Short side | Refine iter | Completed | Approx tracking FPS | Conservative pose failure marker |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| A1 | 0 | 480 | 1 | 300/300 | 13.00 | CAD projected center first leaves image at frame 280 |
| A2 | 0 | 480 | 3 | 300/300 | 11.50 | CAD projected center first leaves image at frame 273 |
| B1 | 0 | 640 | 1 | 300/300 | 9.97 | CAD projected center first leaves image at frame 279 |
| B2 | 0 | 640 | 3 | 300/300 | 9.06 | CAD projected center first leaves image at frame 298 |
| C1 | 1 | 640 | 1 | 300/300 | 6.10 | CAD projected center remains inside image for all 300 frames |
| C2 | 1 | 640 | 3 | 300/300 | 5.75 | CAD projected center remains inside image for all 300 frames |

The FPS values use the timestamp span between the first and last saved pose, so they are approximate tracking-only throughput and exclude most initial registration time.

## Objective trajectory observations

Projected CAD-center summaries:

- A1: first `(305.8, 311.3)`, final `(-19.0, 288.5)`, max frame-to-frame center jump 6.4 px.
- A2: first `(305.8, 311.3)`, final `(-13.7, 70.9)`, max frame-to-frame center jump 13.8 px.
- B1: first `(408.4, 415.1)`, final `(-25.3, 380.2)`, max frame-to-frame center jump 9.0 px.
- B2: first `(408.4, 415.1)`, final `(-16.3, 163.7)`, max frame-to-frame center jump 18.9 px.
- C1: first `(408.4, 415.1)`, final `(614.0, 78.7)`, max frame-to-frame center jump 5.7 px.
- C2: first `(408.4, 415.1)`, final `(242.3, 113.0)`, max frame-to-frame center jump 10.1 px.

These center trajectories alone cannot determine the exact first visually obvious drift frame because the real object itself moves. The first frame where the projected CAD center exits the image is therefore used only as a conservative, reproducible failure marker. Exact mask/pose disagreement is evaluated in the later XMem consistency stage.

## Issues found while running baseline

1. RGB-only registration uses an all-zero depth map. With `debug >= 2`, `FoundationPose.register()` attempted to write an Open3D point cloud from zero valid points and crashed in `colors.max()`. The debug-only point-cloud write is now skipped when no valid depth points exist.
2. The DevSpace execution environment is headless. `cv2.imshow()` caused a Qt/xcb abort even though debug images could be saved. `run_demo_without_depth.py` now detects missing `DISPLAY` and skips only the window display while retaining saved visualizations.

## Baseline conclusion

- Increasing short side from 480 to 640 does not eliminate cumulative drift in mode 0.
- Increasing tracking refinement from 1 to 3 delays the conservative out-of-image failure in the 640 mode-0 run from frame 279 to frame 298, but does not solve the drift mechanism.
- Mode 1 now runs through all 300 frames at 640 without a resolution mismatch and keeps the projected CAD center in-frame, but it is substantially slower.
- The next required step remains XMem mask propagation plus mask/pose consistency checking and relocalization.
