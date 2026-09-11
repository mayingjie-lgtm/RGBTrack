# Repository Guidelines

## Project Structure & Module Organization

RGBTrack extends FoundationPose with RGB-only pose tracking and XMem-guided recovery.

- `run_demo_*.py`: runnable demos; `run_demo_without_depth.py` includes the RGB-only recovery pipeline.
- `estimater.py`, `datareader.py`, `tools.py`, and `Utils.py`: pose estimation, input loading, tracking helpers, and shared utilities.
- `xmem_wrapper.py` and `XMem/`: mask propagation integration and bundled upstream implementation.
- `learning/`, `mycpp/`, and `bundlesdf/mycuda/`: learned models and native C++/CUDA extensions.
- `demo_data/`, `weights/`, and `assets/`: demo inputs, model checkpoints, and documentation assets. Generated results belong in `debug/`.

## Build, Test, and Development Commands

Run commands from this repository's root. Follow `readme_original.md` for FoundationPose dependencies and `USAGE_CN.md` for local usage. Inference requires compatible GPU/CUDA dependencies and downloaded checkpoints.

```bash
conda activate rgbtrack
bash build_all_conda.sh  # Rebuild C++ and CUDA extensions.
python run_demo_without_depth.py --mesh_file demo_data/zhuying/mesh/mid_haican_m_simplified.obj --test_scene_dir demo_data/zhuying --mode 0 --use_xmem
```

The demo command runs RGB-only tracking with mask propagation. XMem expects weights at `XMem/saves/XMem.pth` by default. Use `RGBTRACK_MASK_RECOVERY_RESULT.md` for the complete validated recovery configuration.

## Coding Style & Naming Conventions

Match each file's existing indentation and formatting; Python indentation varies across modules. Use `snake_case` functions and variables and `PascalCase` classes. No repository-wide formatter or linter configuration is provided.

Add concise English docstrings to every new or modified Python function/class; prefer Doxygen for C++ functions. Document functionality, main parameters, returns, and necessary restrictions. Explain non-obvious coordinate transforms, units, recovery transitions, and boundary handling. Keep changes focused and avoid unrelated refactoring or dependency edits.

## Testing Guidelines

No unified first-party automated test suite or coverage threshold is defined. Define success criteria before implementation. Reproduce bugs before fixing them, add focused regression cases, and use `test_*.py` for new Python tests.

For tracking changes, follow `RGBTRACK_MASK_RECOVERY_RESULT.md`: validate all 300 frames, non-empty masks, drift recovery, saved poses, and `tracking_metrics.csv`. Compare relevant results with `RGBTRACK_AB_RESULT.md`. Record exact commands, configuration, and observed outcomes; distinguish completed validation from untested assumptions.

## Commit & Pull Request Guidelines

Recent history uses concise subjects with `feat:`, `fix:`, and `test:` prefixes. Keep commits scoped to one change. PR descriptions should state purpose, linked issues when applicable, reproduction commands, validation results, and overlays for visual changes.

Keep downloaded weights and generated debug artifacts out of commits. Clarify ambiguous requirements before coding, prefer simple implementations, and update outdated comments when changing behavior.
