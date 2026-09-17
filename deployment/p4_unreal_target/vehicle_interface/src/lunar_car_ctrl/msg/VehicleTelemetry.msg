# ============================================================
# 车辆遥测数据 (VehicleTelemetry)
# ============================================================
# 由 lunar_car_ctrl 节点从 UE 遥测反馈中解析并发布
#
# UE 原始数据为左手坐标系 (LH): X 前进, Y 向右, Z 向上
# 节点自动转换为右手坐标系 (RH): X 前进, Y 向左, Z 向上
#
# 标准字段 (x/y/z, qx~qw): 右手坐标系 (RH, Y 取反)
# LH 字段 (x_lh~qw_lh):    左手坐标系 (UE 原始数据)
#
# 坐标系: X 前进, Y 向右(LH)/向左(RH), Z 向上
# 车轮编号: FL=前左(0), FR=前右(1), RL=后左(2), RR=后右(3)
# ============================================================

std_msgs/Header header       # stamp=节点当前时间, frame_id="base_link"

# ---- UE 原始位姿 (左手坐标系 LH) ----
float64 x_lh                 # 位置 X (m)
float64 y_lh                 # 位置 Y (m) — 左手系: Y 向右
float64 z_lh                 # 位置 Z (m)
float64 qx_lh                # 四元数 X (左手系)
float64 qy_lh                # 四元数 Y (左手系)
float64 qz_lh                # 四元数 Z (左手系)
float64 qw_lh                # 四元数 W (左手系)

# ---- 转换后位姿 (右手坐标系 RH, Y 轴取反) ----
# 转换公式: x=x_lh, y=-y_lh, z=z_lh
#           qx=-qx_lh, qy=qy_lh, qz=-qz_lh, qw=qw_lh
float64 x                    # 位置 X (m)
float64 y                    # 位置 Y (m) — 右手系: Y 向左
float64 z                    # 位置 Z (m)
float64 qx                   # 四元数 X
float64 qy                   # 四元数 Y
float64 qz                   # 四元数 Z
float64 qw                   # 四元数 W

# ---- 车体速度 (车体坐标系) ----
float64 vx                   # 线速度 X (m/s)
float64 vy                   # 线速度 Y (m/s)
float64 vz                   # 线速度 Z (m/s)
float64 wx                   # 角速度 X (rad/s)
float64 wy                   # 角速度 Y (rad/s)
float64 wz                   # 角速度 Z (rad/s)

# ---- 四轮轮速 (rad/s) ----
float64 wheel_speed_fl       # 前左轮速
float64 wheel_speed_fr       # 前右轮速
float64 wheel_speed_rl       # 后左轮速
float64 wheel_speed_rr       # 后右轮速

# ---- 四轮转向角 (rad) ----
float64 steer_angle_fl       # 前左转角
float64 steer_angle_fr       # 前右转角
float64 steer_angle_rl       # 后左转角
float64 steer_angle_rr       # 后右转角
