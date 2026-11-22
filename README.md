# audio-ambisonic-decode

## 题面

接入一阶 Ambisonic 后，左右声像反了、绕场旋转还会把音量带跑：① `sh_basis` 把 Y/Z 分量交换（Y 用 sin 仰角、Z 用 sin 方位），左右与上下颠倒；② N3D 归一化漏掉 √3/√(3/2) 因子，一阶通道能量偏高 3 倍；③ `to_sn3d` 对 W 也乘了 1/√3，全向通道被衰减；④ `decode_matrix` 用转置（扬声器→声场）当作解码矩阵，方向镜像；⑤ `decode_matrix` 未按 max-rE 加权，高频方向性过强（能量分布断言失败）；⑥ `apply_matrix` 未归一化扬声器增益，总能量随扬声器数量变化；⑦ `rotate` 把 W 也参与旋转，全向分量被调制；⑧ `energy_ratio` 用 W²/总和，忽略 W 本身是能量基准，结果恒 <1/3；⑨ `validate_basis` 只检查通道数不检查正交性（球谐在均匀采样下的点积应为 0）；⑩ `decode_batch` 复用旋转后的声场对象，第二帧从第一帧的旋转状态开始。难点在归一化与矩阵方向：N3D/SN3D 各阶系数不同，解码矩阵必须是声场→扬声器。`ambi/decode.py` 是核心，`tests/test_decode.py` 15个红。要修的是：① 让 Y 随方位角、Z 随仰角，保留 `[W, Y, Z, X]` 顺序；② 恢复一阶的 √3 N3D 因子；③ `to_sn3d` 只缩放一阶、保持 W 不变；④ 按声场→扬声器方向构造解码矩阵，禁止转置；⑤ 应用 `1/√3` 的 max-rE 一阶权重并保持行能量为 1；⑥ `apply_matrix` 按输入能量归一，抵消扬声器数量变化；⑦ 旋转只作用于一阶分量，W 保持不变；⑧ 先把 N3D 一阶能量折回 SN3D 再计算 W 占比；⑨ `validate_basis` 校验均匀采样下的归一化与正交性；⑩ `decode_batch` 每帧都从自己的原始系数独立旋转和解码。

## 复现

在仓库根目录运行：

```console
$ python -m pytest tests -q
15 failed
```

初始状态必须恰好是 `15 failed`：没有 collection error、import error 或 syntax error，15 条测试全部因 `NotImplementedError` 失败。修复后的目标为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`）与 pytest；离线、无随机、无网络、不 `sleep`。
- 角度统一用弧度；坐标系为右手系，`+X` 前、`+Y` 左、`+Z` 上；方位角从 `+X` 逆时针转向 `+Y`，仰角为水平面以上的角度。
- 通道顺序为 ACN 的 `[W, Y, Z, X]`；N3D 下 `W=1`、`[Y, Z, X]=√3·(y, z, x)`，SN3D 的一阶系数再除以 `√3`。
- 正前方单位方向为 `W=1、X=1、Y=Z=0`（测试断言 `X/√3=1`）；对应 N3D 系数是 `W=1、X=√3、Y=Z=0`。
- 解码矩阵是“声场→扬声器”（行=扬声器、列=ACN 通道），每行平方和为 1；max-rE 一阶权重固定为 `1/√3`。
- 断言容差为 `1e-9`；`tests/test_decode.py` 是验收规格，**不可修改**，修复只发生在 `ambi/decode.py`（`ambi/__init__.py` 仅做导出）。
