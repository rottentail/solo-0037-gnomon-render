# Gnomon 真半透明实现笔记

## 设计决策
- `Surface.opacity`（默认 1.0，[0,1]）是唯一连续半透明控制；`alpha_cutoff` 保留二值镂空语义。
  有效 alpha = 纹理二值遮罩 × opacity。
- 两遍光栅化：pass1 不透明/镂空走基线最近深度；pass2 半透明片元写入每子采样固定 8 层 A-buffer。
- 合成在 resolve 阶段按深度稳定排序后做精确 float64 Porter-Duff source-over（前向累加，
  背景与不透明获胜面各经剩余 carry 进入一次），结果与节点排列无关；等深片元按遍历顺序。
- 拾取：最近不透明面优先，前层半透明不阻挡，through_ids 按前到后列出；无 opaque 时回退最近半透明层。

## 边界
- MAX_TRANSLUCENT_LAYERS=8，LAYER_RECORD_BYTES=96，半透明帧每子采样 896 字节，不透明保持 128。
- 超层丢弃并计入 stats.translucent_overflow_samples；160MiB 预算在 preflight/render 拒绝。

## 修复历程
1. 半透明深度测试误用严格小于 → 前层片元被丢弃，改为 `z < depth+eps`。
2. 跨行 slice shift 别名导致层数据损坏 → 改逐格 `[pos+1:]=[pos:-1]` 旋转。
3. argsort 把 id=0 空槽排到前面 → 空槽深度为 inf，共享深度序自然置后；through_ids 再过滤 id>0。
4. 无 opaque 回退选了最深层 → 取排序后 slot 0（最近）。
5. 合成背景被 carry 重复加权 → 累加从黑+carry=1 开始，背景仅在末尾经 carry 进入一次。
   已用独立 back-to-front over 参考验证到 1e-12。

## 验证
- 十条生产 flow 全部单独通过；translucency-proof 八项断言；两个 examples 脚本；HTTP walkthrough。
- 区域裁切所有通道逐元素相等；samples=1/2/3；opacity=0 不可见不可选；schema 边界拒绝。
