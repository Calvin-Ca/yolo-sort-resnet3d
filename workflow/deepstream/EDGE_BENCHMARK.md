# Jetson 边缘设备端到端测试

## 测试环境

- 设备：NVIDIA Jetson Orin NX Developer Kit
- JetPack/L4T：5.1.1 / 35.3.1
- CUDA：11.4
- TensorRT：8.5.2
- DeepStream：6.2.0
- 输入：H.264，2490×1400，10 FPS，301 帧（约 30.1 秒）
- YOLO 引擎：`worker.640.yolov5n6.fp16.engine`
- Action 引擎：`action_resnet3d_t30_s200.fp16.engine`

## 端到端结果

| 管线 | 帧数 | 检测数 | 跟踪目标数 | 耗时 | 端到端 FPS |
|---|---:|---:|---:|---:|---:|
| YOLO + NvSORT 基线 | 301 | 1387 | 6 | 8.38 s | **35.93** |
| YOLO + NvSORT + 3D Action（同步） | 301 | 1386 | 6 | 13.11 s | **22.96** |
| Action `classifier-async-mode=1`（含 EOS 收尾） | 300 | 1382 | 6 | 14.44 s | **20.77** |
| 时序 `subsample=1`（隔帧采样） | 301 | 1387 | 6 | 13.53 s | **22.25** |

同步 3D Action 测试输出：

```text
benchmark/deepstream/native_fixed.mp4
benchmark/deepstream/native_fixed.jsonl
benchmark/deepstream/native_fixed.actions.jsonl
benchmark/deepstream/native_fixed.summary.json
```

## 结论

object_id 时序键和 30 帧缓存已经正常工作，`roi_restore_misses=0`。当前
端到端瓶颈主要在每帧多目标 ROI 裁剪、缩放、归一化及 GPU 时序缓存写入；
异步模式需要 EOS 收尾等待，结果完整但性能下降；仅跳过缓存帧写入也没有提升性能。后续应让非采样帧
直接跳过 ROI 预处理，或减少 ROI 空间尺寸，并用 DeepStream/Nsight 分阶段测量。
