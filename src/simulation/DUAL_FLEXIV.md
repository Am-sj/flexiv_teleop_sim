# 双臂非夕 Rizon4 仿真装配

运行环境：`FlexivDualArm-v1`。使用提供的
`非夕双臂立柱机械臂底座-20260523.stp` 底座，安装两台已有的
Rizon4 + Grav（Flexiv-GN01）机械臂。

## 启动

在已安装项目仿真依赖的 Python 环境中运行：

```bash
cd src/simulation
python dual_flexiv_viewer.py
```

程序启动后，两臂保持接近参考照片的准备姿态：上臂向两侧展开，肘部在外侧，
前臂向前收，夹爪向下，夹爪张开。Ctrl+C 退出。
这份姿态根据单张照片近似调整，不是从实机读取的精确关节角；
可继续调整，不影响机械臂的固定安装位姿。

默认关节角（度，joint1 至 joint7）：

| 左臂 | 右臂 |
| --- | --- |
| `[-18.5, 112.5, 31.7, 129.7, 140.4, 19.8, 0]` | `[10.3, 114.2, -40.8, 146.8, -148.6, 44.3, -77.2]` |

右腕按原图中靠近镜头那条机械臂的可见方向调整，不采用左腕镜像：
侧面关节盖到腕筒的连接向画面右上方抬起，腕筒向右下方倾斜，
夹指的展开方向呈斜向。侧面关节盖朝向观察者，避免向上翻转。
右臂前臂绕轴姿态与腕部配合调整，肩、肘和腕中心位置保持接近原布局；
左臂姿态保留。单张照片不能唯一确定实机三维关节角，以上是可见构型的近似复现。

配置文件 `assembly.json` 中的 `arm_qpos` 使用弧度。

无窗口运行 CPU 物理验证：

```bash
python dual_flexiv_viewer.py --headless --duration 2
```

输出渲染图片（仍需可用的 Vulkan/GPU）：

```bash
python dual_flexiv_viewer.py --headless --duration 1 --output /tmp/dual_flexiv.png
```

## 坐标和安装方向

**左右按机器人自身朝前时定义**：世界坐标 `+X` 为前方，`+Y` 为左侧，
`+Z` 向上。从正面看机器人时，画面左右与机器人的左右相反。

底座 CAD 的单位为毫米，仿真统一为米。转换保留原始的 36 个实体，
把中央立柱轴线放在世界原点上方，并使最低脚轮接触地面。
底座作为静态物体固定，脚轮没有驱动或滚动关节。

| 项目 | 左臂 | 右臂 |
| --- | --- | --- |
| CAD 安装盘中心 / mm | `(226.95, -132.5, 1210)` | `(-273.05, -132.5, 1210)` |
| 仿真基座位置 / m | `(0.115, 0.250, 1.305)` | `(0.115, -0.250, 1.305)` |
| 基座四元数 / wxyz | `(-0.5, 0.5, -0.5, 0.5)` | `(0.5, 0.5, 0.5, 0.5)` |
| 基座安装轴方向 | 世界 `+X` | 世界 `+X` |
| 航插方向 | 向下，世界 `-Z` | 向上，世界 `+Z` |
| 动作字典键 | `flexiv_rizon4-0` | `flexiv_rizon4-1` |

两安装盘相距 **500 mm**，法向均朝前。原始 CAD 安装高度为 1210 mm；
落地变换包含 95 mm 的脚轮高度偏移，因此世界安装高度为 1305 mm。
Rizon4 基座网格上的航插开口位于其局部 `+Y` 侧，安装面法向为局部 `+Z`。
左右基座旋转分别把这两个方向映射到上表所列世界方向。

安装数据、默认准备姿态、源文件 SHA256 和网格尺寸记录于：
`mani_skill/assets/scenes/flexiv_dual_arm/assembly.json`。
其中 `mounts.left/right.arm_qpos` 只控制准备姿态，`position` 和
`quaternion_wxyz` 控制固定装配位姿。

## 独立控制两臂

每台机械臂保持独立的关节、控制器和碰撞模型。位置控制时，每臂动作
包含 7 个绝对关节角（弧度）和 1 个夹爪命令；`+1` 张开，`-1` 闭合。
两台机械臂共 28 个活动关节（包括夹爪联动和虚拟宽度关节），动作共 16 维，
以左右两个字典条目传入，而不是一个 16 维数组。

```python
import gymnasium as gym
import mani_skill.envs
from dual_flexiv_viewer import hold_actions

env = gym.make("FlexivDualArm-v1", control_mode="pd_joint_pos", render_mode="human")
env.reset(seed=0)
actions = hold_actions(env)
actions["flexiv_rizon4-0"][0] += 0.1  # 只改变左臂第一个关节目标
actions["flexiv_rizon4-1"][7] = -1.0  # 只闭合右臂 Grav
for _ in range(40):
    env.step(actions)
    env.render()
env.close()
```

该装配也支持现有的 `pd_joint_delta_pos`、`pd_ee_delta_pose` 控制模式；
它们的单臂动作维度分别为 8、7。`hold_actions` 仅用于 `pd_joint_pos`。
现有 `teleop_sim.py` 保持单臂串口遥操作流程；`dual_flexiv_viewer.py`
只发送保持姿态的目标，不读取串口。两套 USB UArm 的遥操作入口为 `dual_teleop_sim.py`。


## 双主臂遥操作与映射调整

```bash
python dual_teleop_sim.py --scene Empty-v1 \
  --left-port /dev/ttyUSB0 --right-port /dev/ttyUSB1 \
  --mapping-mode joint
```

`--scene FlexivDualArm-v1` 加载立柱和双臂；`--scene Empty-v1` 保留同样的
左右固定安装位姿及初始关节姿态，但不加载立柱。空场景也不包含立柱的碰撞体。
左右主臂各用一个串口，每套读 `000–007`；两侧摆好参考姿态、释放扳机并保持不动，
在终端按一次 Enter 请求两侧标定，两侧完整成功后才推进遥操作仿真。
标定只采集角度参考，不识别关节顺序、正负方向或基座安装旋转。
默认逐帧输出 `[left] Servo angles (degrees): [...]` 和对应的 `[right]` 行，
与单臂输出同样的标定后偏移（度），数组位置 0–7 对应舵机 000–007。
可用 `--no-debug-angles` 关闭输出，原 `--debug-angles` 命令仍可使用。
`--rate` 默认 50 Hz，同时控制采集及消费循环的目标频率；无新数据不推进仿真。
串口、PWM、扳机阈值、控制后等待和默认材质参数与单臂对齐，完整参数见
[README.md](README.md#双-uarm--双臂-rizon4)。

默认 `joint` 模式保持与单臂一致的关节位置控制策略：

```text
q_target[side] = clip(q_initial[side] + signs[side] * scales[side]
                     * radians(current_degrees - calibration_degrees)[indices[side]],
                     joint_lower, joint_upper)
```

每侧用自己的初始角度、限位和夹爪状态；每个仿真关节的输入集中在
`dual_teleop_sim.py` 顶部 `ARM_MAPPING["left"]` 和 `ARM_MAPPING["right"]`：

| 字段 | 含义 |
| --- | --- |
| `servo_indices` | 按仿真 joint1–joint7 顺序填写输入通道 0–6，每个通道恰好出现一次 |
| `joint_signs` | 按仿真 joint1–joint7 顺序填写 `+1` 或 `-1` |
| `joint_scales` | 按仿真 joint1–joint7 顺序填写角度比例，先保留 1 |

当前左右通道顺序均为 `[0,1,2,3,4,5,6]`，方向均沿用单臂 `[1,-1,1,1,1,1,1]`。
硬件排查交流使用从 0 开始的关节编号：编号 0–6 对应 URDF 名称 joint1–joint7，
例如编号 1、3、5 对应 joint2、joint4、joint6。列表索引与从 0 开始的编号一致。
通道 `007` 独立控制夹爪，不参与机械臂通道交换。

按以下步骤确认硬件映射：

1. 标定成功后先固定右主臂，每次只小幅移动左主臂的一个物理关节。
2. 查看 `[left] Servo angles (degrees): [...]`，记录主要变化的舵机 ID、偏移正负，
   以及从臂实际转动的关节。其余主臂关节尽量保持不动。
3. 按机械结构的对应关系确认期望驱动的从臂关节；如果通道对应错了，修改
   `servo_indices`。例如已确认舵机 001 应控制 joint3、002 应控制 joint2，
   才将其改为 `[0,2,1,3,4,5,6]`。
4. 通道对应正确、同一个关节转向相反时，只翻转对应 `joint_signs` 一项。
   例如要反向 joint3，就翻转列表第三项，而不是舵机编号所在位置。
5. 用相同方法检查右主臂，单独修改右侧配置。每次修改后重启并重新标定。

记录格式：`侧别 / 移动的物理关节 / 变化的舵机 ID / 偏移正负 / 实际从臂关节 / 期望关节`。
屏幕左右受相机影响，空间方向按世界 +X 前、+Y 左、+Z 上判断。
如果转动的是正确对应关节，只是末端在世界坐标中的运动方向不同，不能仅靠正负号修正轴向。
这时需检查主臂与从臂的安装和参考姿态，并明确是否要改用末端空间跟随。
不要为了修正映射改动 `position`、`quaternion_wxyz`；`arm_qpos` 只在需要
调整准备姿态时修改，修改后应重新验证姿态、关节限位和碰撞。

可选 `--mapping-mode single-world` 是额外的单臂 TCP 运动参考 FK/IK 策略，
会协调多个从臂关节，不属于默认的逐关节策略。它不替代上述通道核对，
也不执行碰撞路径规划。

## 网格与碰撞

底座视觉网格来自 STEP 曲面三角化，静态碰撞使用同一几何的三角网格。
没有用整体凸包代替底座，以免把立柱周围和安装盘之间的空隙填满。
两臂之间、机械臂与底座之间均保留碰撞。
每个夹爪的内部连杆使用独立碰撞过滤位，保留左右夹爪之间的碰撞。
`rizon4.srdf` 过滤腕部与夹爪刚性安装接缝的一对相邻碰撞网格，
消除该接缝原有的约 1 mm 几何重叠产生的内力。

运行时不需要 CAD 软件、gmsh 或原始 STP 文件。原始 STP 和参考照片不随仓库发布，
需要重新生成网格时请自行提供源 STP 文件；
在开发环境安装 `gmsh`、`trimesh`、`scipy`，从仓库根目录运行：

```bash
python src/simulation/mani_skill/assets/scenes/flexiv_dual_arm/import_stand.py \
  非夕双臂立柱机械臂底座-20260523.stp
```

测试：

```bash
cd src/simulation
python -m unittest test_dual_flexiv test_flexiv_rizon4 test_teleop_sim test_dual_teleop_sim
```
