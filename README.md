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

服务运行后可执行 `.venv/bin/python examples/http_walkthrough.py`，通过真实 HTTP 上传样例并下载静帧。

## 成像约定

- 右手世界坐标；Y 向上，相机看向 target。旋转单位为度，局部顺序为 X、Y、Z，随后由父节点变换进入世界空间。
- 透视 fov 为垂直视角，正交 span 为垂直世界长度；near/far 为正距离。六个齐次平面裁剪后进行透视校正插值。
- PNG 纹理按未附 ICC 的 sRGB 解码，UV 原点在左下，双线性采样。repeat/clamp 均可。alpha_cutoff 是二值裁切，不表示半透明混合。
- 材质基色为 sRGB，着色和抗锯齿平均在线性空间，PNG 最后编码回 sRGB；超出显示范围的亮度仅在 PNG 截断，NPZ 保留。
- 生成法线为相邻三角形面积加权；可显式提供逐顶点法线。非均匀缩放使用逆转置法线变换。
- 单面剔除按变换后的顶点绕序；镜像会翻转绕序。双面不会自动把法线朝向相机。共面等深片元保留场景遍历时先出现的表面。
- samples=1/2/3 是每轴均匀采样。颜色是全体子采样含背景的平均；拾取、深度、法线、世界位置来自最近的有覆盖子采样。对象标识为场景节点序号，0 为背景。
- 平面镜：surface.mirror 把单平面网格节点标记为镜面。reflectance 为反射占比（其余为自身着色）；include/exclude 按节点键控制哪些物体入镜（exclude 优先，镜面自身永不入镜）。反射面朝向当前渲染镜头，从任一侧观察都得到该视角的反射；镜中镜按普通表面着色，不产生二次反射。
- 反射不是贴图：每个镜面按镜像相机（主相机关于镜面平面的镜像）重渲场景，镜后几何按半空间裁剪剔除，因此镜后物体不会出现在镜内，画面外物体可以出现在镜内。反射图与主画面共用 NDC，镜面片元的反射采样落在自身屏幕位置，透视、镜边与拾取逐像素对应。
- 渲染请求 reflection.resolution_scale（默认 1.0，≤1）降低反射缓冲分辨率以控制成本；reflection.enabled=false 完全关闭反射（镜面只显示自身着色，不产生反射通道）。
- 含镜面时 channels.npz 增加 reflection_id（该像素反射到的节点序号，0 为无）与 reflection_world（反射表面世界坐标），拾取接口在镜面像素上同时返回镜面自身与反射目标。
- region=[x,y,w,h] 使用整图左上坐标和原始镜头，不改变构图。输出像素坐标是区域内坐标；拾取另返回全图子采样位置。
- 深度 NPZ 是 OpenGL NDC [-1,1]，背景为 +inf；depth.png 每张按自身有效深度范围做可读预览，不能当作物理距离或跨图绝对标尺。

## 工作流

| 入口 | 结果与边界 |
|---|---|
| POST /v1/meshes | 三角索引、面积、法线、UV、包围盒校验，退化面拒绝 |
| POST /v1/textures | PNG 解码、直通透明度、尺寸与格式约束 |
| POST /v1/scenes | 唯一键、父图、资源和相机校验，镜面平面性校验，不可变场景 |
| POST /v1/scenes/{id}/revisions | 完整新修订，predecessor 追溯，不覆盖前版本 |
| POST /v1/preflight | 世界包围盒、可见三角数、反射缓冲与预计帧缓冲、可受理判断 |
| POST /v1/renders | 冻结场景和实际资源，生成五张 PNG、channels.npz、metadata.json；镜面按请求预算生成反射通道 |
| GET /v1/renders/{id}/pixels/{x}/{y} | 最近子采样表面及其世界坐标，背景返回空表面 |
| POST /v1/comparisons | 相同镜头、尺寸、区域与采样规则下，比较线性颜色、轮廓、语义节点及深度 |
| POST /v1/sheets | 绕目标的多个 Y 轴视角，整合 sheet.png 与 index.json |
| POST /v1/renders/{id}/bundles | 自描述 ZIP，包含资源输入、通道语义及逐文件摘要 |
| POST /v1/resources/{id}/state/retired | 拒绝新场景绑定，旧场景继续引用；active 可恢复 |
| POST /v1/scenes/{id}/state/archived | 阻止新成像；active 可恢复，已有产物可读 |

## 环境和资源边界

每个网格最多 20,000 顶点 / 30,000 三角，场景最多 100 节点 / 60,000 可见三角。纹理最多 1024×1024，编码 PNG 最多 4 MB。成像边长 16–1024，保守估算采样缓冲不超过 160 MiB；这不是整个 Python 进程的内存限制，解码资源、JSON、数组临时副本和归档还需要内存。静帧没有阴影、透射、骨骼、动画或外部模型格式导入。

`.gnomon-data/gnomon.sqlite3` 保存记录，artifacts/ 保存校验摘要的结果。运行产物不进入 Git。应使用同一目录和单个 worker；不承诺跨进程互斥或分布式执行。HTTP 是可信环境中的成像接口，不含身份系统；不要直接暴露到公共网络。

## 验证与测试边界

`gnomon.demo` 是生产 API 的有界操作演示，`--flow` 可单独运行十条工作流。它实际生成与读取产物，展示正常和拒绝路径。自动化测试套件尚未建立（testing: deferred），本基线不包含 tests、测试框架和 test_command。

`examples/mirror_showcase.py` 是平面镜反射的可生成示例：倾斜镜面、前方遮挡、贴近镜边物体、画面外物体反射与镜后遮挡，输出 full/low/off/limited 四种渲染、对比热图和实测报告（report.md），并逐像素校验反射不变量。运行：

```sh
.venv/bin/python examples/mirror_showcase.py --output run-output/mirror-showcase
```
