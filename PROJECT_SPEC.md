# Gnomon 产品与实现契约

用户是需要可追溯静帧与多视角校样的三维插图制作者。服务拥有网格、纹理、节点层级场景、镜头、材质、定向光、渲染记录、比较记录、多视角记录和成像包。它计算从模型坐标到像素表面的对应关系，而不是仅登记成像请求。

网格和纹理内容不可变，active/retired 控制未来引用；场景新修订保留 predecessor，active/archived 控制未来成像；渲染 queued→running→complete/failed，多视角 building→complete/failed，对比 measuring→complete/failed，归档 packing→complete/failed。重启时中间状态转 failed，已完成内容不受影响。HTTP 请求同步，状态查询用于持久回溯。

## 职责及事实流

schema 负责公共数据形状和局部约束；mesh 负责拓扑和法线；texture/color 负责颜色编码与透明度；catalog 拥有不可变输入和受保护的状态转换；scene 负责父图、世界矩阵、可见性和资源绑定；camera 负责镜头空间；clipping 负责齐次裁剪与属性交点；raster 负责样本覆盖和最近深度；shading 负责材质光照；channels 负责不同通道的归约语义；inspection 解释最近子采样；comparison 仅对语义相容的成像量化；sheet 组织多镜头的空间布局；artifacts/bundle 拥有产物摘要和交付语义；service/api 组织同步操作和 HTTP 错误。

共享不变量包括：裁剪产生的新顶点属性仍对应原始表面；深度获胜者与对象、法线和世界位置一致；颜色平均与拾取选择有明确不同语义；同镜头区域结果与整图裁切一致；资源停用不改变旧修订含义；比较中的数值节点编号不充当跨修订语义身份；成像包描述实际使用输入而非当前资源状态。

十条有界操作及源码模块在 .project-manifest.json；可直接执行的生产操作在 .project-runtime.json。每项同时覆盖有意义的失败或替代情况。examples/ 中提供可生成的网格和 PNG，所有运行可在已安装依赖后无网络完成。

## 失败边界

非法索引、零面积面、无效法线/UV、父环、失效引用、奇异变换、无效相机轴、错误 PNG、纹理缺少 UV、超预算请求、镜头不一致比较、超出像素、摘要损坏都返回领域失败。渲染开始后的失败保留 failed 记录，API 返回错误。原始异常只用于可信环境排障。本版本不承诺超时取消、增量渲染、跨进程资源竞争和实际进程 RSS 硬限制。

## 可复现性

Python >=3.11，依赖精确固定。NumPy CPU 使用 float64，无 GPU、字体或系统编解码器依赖；Pillow 负责 PNG 和多视角标签。ZIP 时间与数据库 ID 不保证二进制可复现，冻结输入下的数值成像可比较。代码未加入测试套件，测试延后；构建与十条生产走查是本次验证范围。
