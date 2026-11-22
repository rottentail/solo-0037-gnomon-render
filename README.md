# Gnomon 静帧成像服务

Python CPU 三角网格成像后端。输入 JSON 网格、PNG 纹理和有层级的场景，输出颜色、深度、世界法线、对象标识、覆盖率及采样世界坐标。用途是三维静态插图与材质校样，不需要 GPU 或外部服务。

## 启动

需要 Python 3.11+。本批实际使用 Python 3.14 / macOS arm64。

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[examples]'
.venv/bin/python -m compileall -q gnomon
.venv/bin/python -m gnomon.demo --output run-output
GNOMON_DATA=.gnomon-data .venv/bin/uvicorn gnomon.api:app --host 127.0.0.1 --port 8737 --workers 1
```

打开 http://127.0.0.1:8737/docs 浏览 OpenAPI。服务仅设计为单进程；CPU 成像请求同步返回，完成/失败状态持久化。目录由服务独占，重启将中断的未完成记录标记 failed；重新提交可生成新结果。

服务运行后可执行 `.venv/bin/python examples/http_walkthrough.py`，通过真实 HTTP 上传样例并下载静帧。`examples/stereo_observation.py` 与 `gnomon.demo --flow stereo-pair` 会生成左右眼像对、逐像素对应图和交叉核验材料。

## 成像约定

- 右手世界坐标；Y 向上，相机看向 target。旋转单位为度，局部顺序为 X、Y、Z，随后由父节点变换进入世界空间。
- 透视 fov 为垂直视角，正交 span 为垂直世界长度；near/far 为正距离。六个齐次平面裁剪后进行透视校正插值。
- PNG 纹理按未附 ICC 的 sRGB 解码，UV 原点在左下，双线性采样。repeat/clamp 均可。alpha_cutoff 是二值裁切，不表示半透明混合。
- 材质基色为 sRGB，着色和抗锯齿平均在线性空间，PNG 最后编码回 sRGB；超出显示范围的亮度仅在 PNG 截断，NPZ 保留。
- 生成法线为相邻三角形面积加权；可显式提供逐顶点法线。非均匀缩放使用逆转置法线变换。
- 单面剔除按变换后的顶点绕序；镜像会翻转绕序。双面不会自动把法线朝向相机。共面等深片元保留场景遍历时先出现的表面。
- samples=1/2/3 是每轴均匀采样。颜色是全体子采样含背景的平均；拾取、深度、法线、世界位置来自最近的有覆盖子采样。对象标识为场景节点序号，0 为背景。
- region=[x,y,w,h] 使用整图左上坐标和原始镜头，不改变构图。输出像素坐标是区域内坐标；拾取另返回全图子采样位置。
- 深度 NPZ 是 OpenGL NDC [-1,1]，背景为 +inf；depth.png 每张按自身有效深度范围做可读预览，不能当作物理距离或跨图绝对标尺。
- 立体像对使用平行离轴相机：双眼共享同一 forward/up 轴，沿图像右向平移半个基线；主点按汇聚距离偏移，因此汇聚面上的视差为零，绝不靠让相机内转（toe-in）实现。基线 `interocular` 与汇聚距离 `convergence` 必须带单位（mm/cm/m/in），`world_unit` 声明一个场景坐标对应多少米；基线必须小于汇聚距离且汇聚面位于 near/far 之间，正交镜头直接拒绝，因为正交的双眼图只会是平移裁图。
- 视差以整图像素计，`right_x - left_x`：负为交叉（比汇聚面近），正为非交叉（更远）；`right_y - left_y` 为竖向视差，平行机位应恒为 0。深度为沿机位前向轴、以基线中点为原点的场景单位。
- 主眼（左眼）仍写出标准五通道 PNG、`channels.npz` 和 `metadata.json`；像对额外写出右眼通道、`pair.png`、`anaglyph.png`、`parallax.png`、`occlusion-mask.png`、`correspondence.png`、`depth-probes.png`、`correspondence.npz`、`stereo.json` 和 `crosscheck.json`。`POST /v1/renders` 的同一 `region` 对双眼生效，仍等于各自整图裁切；`metadata.json` 的 `summary` 直接给出基线米数、汇聚来源、竖向视差是否达标、视差是否随深度变化、可信像素比例，以及每个无可信对应状态的位置与外接框。

## 工作流

| 入口 | 结果与边界 |
|---|---|
| POST /v1/meshes | 三角索引、面积、法线、UV、包围盒校验，退化面拒绝 |
| POST /v1/textures | PNG 解码、直通透明度、尺寸与格式约束 |
| POST /v1/scenes | 唯一键、父图、资源和相机校验，不可变场景 |
| POST /v1/scenes/{id}/revisions | 完整新修订，predecessor 追溯，不覆盖前版本 |
| POST /v1/preflight | 世界包围盒、可见三角数、预计帧缓冲与可受理判断 |
| POST /v1/renders | 冻结场景和实际资源，生成五张 PNG、channels.npz、metadata.json；带 `stereo` 时另生成双眼像对与核验材料 |
| GET /v1/renders/{id}/pixels/{x}/{y} | 最近子采样表面及其世界坐标，背景返回空表面 |
| GET /v1/renders/{id}/correspondence/{side}/{x}/{y} | 左或右某一像素在另一眼的对应像素、视差、深度、状态与可信判断 |
| POST /v1/comparisons | 相同镜头、尺寸、区域与采样规则下，比较线性颜色、轮廓、语义节点及深度；立体记录以左眼主通道参与，且只与另一立体记录比较 |
| POST /v1/sheets | 绕目标的多个 Y 轴视角，整合 sheet.png 与 index.json |
| POST /v1/renders/{id}/bundles | 自描述 ZIP，包含资源输入、通道语义及逐文件摘要 |
| POST /v1/resources/{id}/state/retired | 拒绝新场景绑定，旧场景继续引用；active 可恢复 |
| POST /v1/scenes/{id}/state/archived | 阻止新成像；active 可恢复，已有产物可读 |

## 立体像对的对应与遮挡

每个方向（左→右、右→左）对源眼每个有覆盖的像素把世界采样点投影到另一眼，得到目标像素并比较两眼的节点、世界点与深度。状态码：

| 状态 | 含义 |
|---|---|
| `matched` | 同一节点、同一世界点、深度在两侧像素足迹内一致，且两侧采样满覆盖——唯一可信状态 |
| `occluded_in_other_eye` | 目标像素在另一眼被更近的表面占据 |
| `surface_mismatch` | 另一眼在同像素存的是不同节点、不同世界点或更近的同节点表面 |
| `no_destination_surface` | 投影点落在另一眼的背景 |
| `outside_destination_frame` | 投影点离开另一眼画幅 |
| `partial_coverage` | 对应成立，但至少一侧是轮廓像素、采样只有部分覆盖 |

`correspondence.npz` 为每个方向保存 `status`、`parallax_pixels`、`vertical_pixels`、`destination_pixel`、`distance_scene_units` 与 `occluder_object_id`，可直接逐像素读取而不是依赖查询接口；`depth-probes.png` 在左眼标注若干分位深度的探针，标签为该像素实测/解析视差与场景深度。

`crosscheck.json` 给出独立核验：两眼的 view 旋转差必须为 0、实测基线与声明一致、汇聚点的左右像素差在 ¼ 像素内；一组按深度分位的解析探针用 `focal_px * baseline * (1/convergence - 1/depth)` 预测视差并与实测比对；线性拟合的视差-深度斜率必须为非零，若只是把单张图平移裁切，斜率会恒为 0。`correspondence.png` 把无可信对应的左眼像素标红，`occlusion-mask.png` 按状态着色，`parallax.png` 红为非交叉、蓝为交叉、灰为不可信。

## 环境和资源边界

每个网格最多 20,000 顶点 / 30,000 三角，场景最多 100 节点 / 60,000 可见三角；立体请求按两只眼的采样缓冲合计计入预检预算。纹理最多 1024×1024，编码 PNG 最多 4 MB。成像边长 16–1024，保守估算采样缓冲不超过 160 MiB；这不是整个 Python 进程的内存限制，解码资源、JSON、数组临时副本和归档还需要内存。静帧没有阴影、透射、骨骼、动画或外部模型格式导入。

`.gnomon-data/gnomon.sqlite3` 保存记录，artifacts/ 保存校验摘要的结果。运行产物不进入 Git。应使用同一目录和单个 worker；不承诺跨进程互斥或分布式执行。HTTP 是可信环境中的成像接口，不含身份系统；不要直接暴露到公共网络。

## 验证与测试边界

`gnomon.demo` 是生产 API 的有界操作演示，`--flow` 可单独运行十一条工作流。它实际生成与读取产物，展示正常和拒绝路径。自动化测试套件尚未建立（testing: deferred），本基线不包含 tests、测试框架和 test_command。
