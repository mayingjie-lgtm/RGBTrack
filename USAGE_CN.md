# RGBTrack 项目使用说明

## 项目用途

RGBTrack 是一个基于 Enhanced FoundationPose 的物体 6D 位姿估计与连续跟踪项目。它根据目标物体的 CAD 网格、相机内参、RGB 图像序列和首帧目标掩码，计算物体在每一帧中的三维位置与姿态。

项目当前主要用于：

- 在 CAD 模型尺寸准确时，仅使用 RGB 图像完成位姿初始化和后续跟踪；
- CAD 模型尺寸未知时，结合首帧深度或单目估计深度恢复模型尺度；
- 结合 XMem 分割结果检测遮挡或跟踪丢失，并在目标重新出现后恢复位姿；
- 为机器人抓取、增强现实叠加和物体运动分析提供逐帧位姿结果。

当前仓库关闭了可选的 TensorRT 模型导入，默认使用 PyTorch 和 CUDA 执行推理。

## 输入与输出

默认的无深度示例需要以下输入：

| 输入 | 说明 |
| --- | --- |
| CAD 网格 | 目标物体的 OBJ 等三维模型；模型尺度应与真实物体一致 |
| `cam_K.txt` | 3×3 相机内参矩阵 |
| `cam_D.txt`（可选） | OpenCV pinhole 畸变参数 `k1 k2 p1 p2 k3`，一行恰好五项 |
| `rgb/*.png` | 按文件名排序的 RGB 图像序列 |
| `masks/*.png` | 与首帧 RGB 同名的目标二值掩码 |

示例数据建议采用以下目录结构：

```text
demo_data/mustard0/
├── cam_K.txt
├── cam_D.txt              # 可选
├── mesh/
│   ├── textured_simple.obj
│   ├── textured_simple.obj.mtl
│   └── texture_map.png
├── rgb/
│   └── <frame_id>.png
└── masks/
    └── <first_frame_id>.png
```

存在 `cam_D.txt` 时，`YcbineoatReader` 会以 `alpha=0` 自动去畸变，并对 RGB、目标 mask、深度和遮挡 mask 使用同一映射。输出分辨率保持 Reader 的目标尺寸；此时 `reader.K` 是去畸变后图像对应的新内参，而 `cam_K.txt` 始终保留相机的原始标定内参。没有 `cam_D.txt` 时保持原有缩放行为。

程序默认将结果写入 `debug/`：

- `debug/ob_in_cam/<frame_id>.txt`：每帧的 4×4 物体到相机坐标系位姿矩阵；
- `debug/track_vis/`：当 `--debug 2` 时保存带位姿框和坐标轴的可视化图像；
- 当 `--debug` 大于等于 1 时，在窗口中实时显示跟踪效果。

> 启动演示程序时会清空 `--debug_dir` 指定目录中的已有内容。需要保留的历史结果应提前复制到其他目录。

## 环境和数据准备

1. 按照 [FoundationPose 原始安装文档](./readme_original.md) 配置 CUDA、Python 依赖并编译扩展。
2. 将姿态细化和评分网络权重放到以下位置：

   ```text
   weights/2023-10-28-18-33-37/model_best.pth
   weights/2024-01-11-20-02-45/model_best.pth
   ```

3. 将演示数据解压到 `demo_data/`。

`weights/`、`demo_data/`、`debug/` 和本地构建目录已被 Git 忽略，不会随源码提交或推送。

## 运行无深度示例

使用仓库自带的默认路径运行：

```bash
python3 run_demo_without_depth.py
```

使用自己的模型和图像序列：

```bash
python3 run_demo_without_depth.py \
  --mesh_file ./demo_data/my_object/mesh/model.obj \
  --test_scene_dir ./demo_data/my_object \
  --debug_dir ./debug
```

常用参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--est_refine_iter` | `5` | 首帧位姿估计的细化次数 |
| `--track_refine_iter` | `1` | 后续每帧跟踪的细化次数 |
| `--debug` | `1` | `0` 关闭可视化，`1` 窗口显示，`2` 同时保存图像 |
| `--debug_dir` | `./debug` | 位姿矩阵和调试结果的输出目录 |
| `--mode` | `0` | `0` 使用零深度跟踪，`1` 使用 CAD 模型渲染的深度跟踪 |
| `--shorter_side` | `480` | 将输入短边缩放到指定像素数，限制候选姿态评分的显存占用 |

增加细化次数通常可以提高稳定性，但会降低处理速度。

## 其他演示入口

- `run_demo_unkown_scale.py`：使用首帧真实深度恢复未知 CAD 尺度；
- `run_demo_colacan.py`：使用 ZoeDepth 生成单目伪深度并恢复尺度；
- `run_demo_clearpose.py`：在 ClearPose 数据上结合 XMem 实现遮挡后的跟踪恢复。

这些入口需要各自的数据集或额外模型，具体说明见[项目主 README](./readme.md)。

## 使用限制

- 当前推理流程依赖 NVIDIA GPU 和 CUDA；
- 首帧必须提供有效的目标掩码，否则无法可靠初始化目标位姿；
- 无尺度恢复时，CAD 模型尺寸误差会直接影响平移和位姿结果；
- 当前配置不加载 TensorRT 后端，不能通过修改 `USE_TRT` 直接启用 TensorRT；
- 本项目输出的是视觉估计结果，机器人实际执行前仍需完成相机标定、坐标系转换和安全检查。
