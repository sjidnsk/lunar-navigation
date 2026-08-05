# Volume 3 月球极区数据源锁摘要

## 结论与边界

2026-08-05 已在仓库外下载并验证 NASA LOLA 南极 DEM、对应 count 栅格和
JAXA LUPEX 六个候选站点资料，并以 seed `4080` 生成固定 split。数据、source lock、
split manifest 和下载临时文件均不进入 Git；本提交只保存非敏感摘要，也没有启动 PPO
训练、生成 checkpoint 或消耗正式 24 小时 GPU 预算。

aggregate source lock 使用相对 `source_root`（`../raw`），本地外部根目录不写入文档。
aggregate 文件 SHA-256 为
`8d422cd9ef478ca15e7e36831ea14e9ba09e9af72565adbf0625a9137b1acab0`，大小为
`16,707` bytes。

## 官方源

| source id | 最终 URL | bytes | SHA-256 |
| --- | --- | ---: | --- |
| `NASA_LOLA_87S_DEM` | `https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif` | 3,465,285,714 | `417a85715406c346e2ecb2fc3abc93d3717121466e3d4950e1a6b977f207b881` |
| `NASA_LOLA_87S_COUNT` | `https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldec_87s_5mpp.tif` | 145,895,726 | `dab531d817b9e7cfddf8ac23ffde9ccfe737efe526e98763ca7ded7a2afcae04` |
| `JAXA_LUPEX_DATA_S1` | `https://zenodo.org/api/records/17153447/files/DataS1.zip/content` | 76,134,049 | `4a2cbb1d9f6ed4a1abd804f9faeee45c6c1f847030b31ba7fb986f280e8ecd77` |

NASA 两个源来自 PGDA product 81（Barker et al. 2021，DOI
`10.1016/j.pss.2020.105119`），使用 NASA reproduction guidance / 美国政府数据语义。
JAXA archive 来自 Zenodo record `17153447`（DOI `10.5281/zenodo.17153447`），
许可为 CC BY 4.0。

## 栅格与站点元数据

NASA DEM 与 count 均为 `40,000 x 40,000`、单波段、5 m 分辨率，覆盖
`[-100000, -100000, 100000, 100000] m`。两者 CRS 为
`Moon2000_spole / Polar_Stereographic`，月球球体半径 `1,737,400 m`，原点纬度
`-90°`，中央经线 `0°`；仿射变换为
`[5, 0, -100000, 0, -5, 100000]`。DEM 为 `float32` 且 NoData 为 NaN；count 为
`uint8` 且 NoData 为 null。verifier 仅把“两侧都为 NaN”视为相同，单边 NaN、CRS、
transform、大小或 SHA 漂移仍会拒绝。

JAXA archive 含 18 个 GeoTIFF，统一使用
`Moon (2015) - Sphere / Ocentric / South Polar` Polar Stereographic CRS，分辨率
为 1 m。六个站点及每站三类成员如下：

| holdout site | DTM | orthomosaic | uncertainty |
| --- | ---: | ---: | ---: |
| `CR1` | 1 | 1 | 1 |
| `GR1` | 1 | 1 | 1 |
| `GR2` | 1 | 1 | 1 |
| `LP1` | 1 | 1 | 1 |
| `MP1` | 1 | 1 | 1 |
| `MP2` | 1 | 1 | 1 |

JAXA 六站点只用于独立 holdout，不进入 NASA train/validation/test。

## 固定 split

split manifest schema 为 `lunar-polar-split-manifest/v1`，seed 为 `4080`。所有 NASA
窗口均为 `1024 x 1024 m`；manifest 共 294 行：

| source | split | 数量 |
| --- | --- | ---: |
| NASA LOLA | train | 192 |
| NASA LOLA | validation | 48 |
| NASA LOLA | test | 48 |
| JAXA LUPEX | holdout | 6 |

内部 `split_sha256` 为
`5d458081972e1ee767c5f91dd5cb42d519214a0111a6e28dfaa7283025ec99e2`，294 行均携带
同一值。split manifest 文件本身大小为 `162,602` bytes，SHA-256 为
`d51b9b824e5e9b2ebe67da70ab9a46c6d4a66489ec8ff063a766347058ba4a2e`。

## 复现命令

以下命令在项目外部 Volume 3 Python 环境中执行；`V3_DATA_ROOT` 必须指向仓库外目录：

```bash
V3_DATA_ROOT="${V3_DATA_ROOT:?set an external Volume 3 data root}"

PYTHONDONTWRITEBYTECODE=1 python \
  training/tools/fetch_polar_data.py \
  --registry training/data_sources/polar_source_registry_v1.json \
  --data-root "$V3_DATA_ROOT/raw" \
  --lock-output "$V3_DATA_ROOT/locks/polar_source_lock_v1.json"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/training/lunar_policy_training"
PYTHONDONTWRITEBYTECODE=1 python \
  -m lunar_policy_training.polar_data.split \
  --source-lock "$V3_DATA_ROOT/locks/polar_source_lock_v1.json" \
  --output "$V3_DATA_ROOT/splits/polar_split_v1.json" \
  --seed 4080
```

重复运行下载命令只允许复用并重新验证已经锁定的文件；中断时保留 `.part`，不得改用旧
DEM。以上 source/split 锁只冻结数据身份和划分，不代表三平台运动能力已经确定，也不解除
正式训练的 capability gate。
