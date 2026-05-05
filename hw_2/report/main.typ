#import "@preview/simple-hust-report:0.1.0": report

#show: report.with(
  logo: image("images/ustc-name.pdf", width: 52%),
  type: "课程实验报告",
  course-name: ("课程名称", "数学建模"),
  title: ("实验题目", "SVD 在图像压缩中的应用"),
  class-name: "少年班学院 2023 级",
  student-id: "PB23000052",
  name: "杨硕",
  instructor: "陈仁杰",
  date: datetime.today().display("[year]年[month]月[day]日"),
  school: "中国科学技术大学",
  header-text: "中国科学技术大学课程实验报告",
  appendix: none,
  bibliography-file: none,
)

= 实验目的

本实验选择作业 2 的 Option 1，研究奇异值分解（Singular Value Decomposition, SVD）在图像压缩中的应用，并在此基础上与离散小波变换（DWT）进行对比。实验的主要目标如下：

- 将图像压缩问题转化为低秩矩阵近似问题；
- 在不调用现成 `SVD` 库函数的前提下，自行实现前若干奇异值与奇异向量的求解；
- 利用截断 `SVD` 对彩色图像进行压缩重建；
- 通过 `PSNR` 和存储比例评价压缩效果；
- 补充实现 `Haar DWT` 压缩方法，并与 `SVD` 进行对比分析；
- 搭建简单 `GUI`，直观观察不同 rank 对压缩效果的影响。

= 数学建模

== SVD 图像压缩模型

将灰度图像表示为矩阵 $A in ℝ^(m x n)$，其奇异值分解形式为

$
  A = U Σ V^T
$

其中 $U$ 和 $V$ 分别为左右奇异向量矩阵，$Σ$ 的对角线上为奇异值
$σ_1 >= σ_2 >= ... >= σ_r > 0$。

若只保留前 $k$ 个最大的奇异值与对应奇异向量，则可得到秩为 $k$ 的近似矩阵

$
  A_k = Σ_(i=1)^k σ_i u_i v_i^T
$

根据 Eckart-Young 定理，$A_k$ 是所有秩不超过 $k$ 的矩阵中，对原矩阵 $A$ 的最佳近似，也就是在 Frobenius 范数意义下使误差最小的 rank-$k$ 近似。

因此图像压缩问题可以建模为：在尽量少保留参数的前提下，用低秩矩阵近似原始图像。

== 彩色图像的通道化建模

对于 RGB 彩色图像，本实验对三个通道分别做 `SVD`：

$
  A^c ≈ A_k^c, c in {R, G, B}
$

最终重建图像由三个通道的近似结果组合得到，分别对应 $A_k^R$、$A_k^G$ 和 $A_k^B$。

若图像尺寸为 $m x n$，则原图每个通道需要存储 $m n$ 个数。对 rank-$k$ 的 `SVD` 表示而言，每个通道只需存储：

- $m k$ 个左奇异向量参数；
- $k$ 个奇异值；
- $k n$ 个右奇异向量参数。

因此三通道总参数量约为

$
  3 k (m + n + 1)
$

对应的存储比例为

$
  rho = k (m + n + 1) / (m n)
$

== DWT 压缩模型

作为对比方法，本实验实现了二维 `Haar DWT`。`DWT` 的核心思想是将图像分解为低频近似分量和高频细节分量，然后只保留绝对值较大的小波系数，从而达到压缩效果。

对图像进行多层二维小波分解后，可以得到四类子带：`LL`、`LH`、`HL` 和 `HH`。

其中 `LL` 表示低频近似信息，`LH`、`HL` 和 `HH` 分别表示水平、垂直和对角高频细节。压缩时保留前若干个重要小波系数，其余系数置零，再通过逆变换重建图像。

= 算法设计

== 手写 SVD

本实验没有调用库中的 `svd`，而是使用“幂迭代 + deflation”的方式逐个提取奇异三元组。对残差矩阵 $R$，重复执行：

1. 随机初始化单位向量 $v$；
2. 交替迭代
   $
     u = (R v) / (‖R v‖)
   $
   $
     v = (R^T u) / (‖R^T u‖)
   $
3. 收敛后计算奇异值
   $
     σ = u^T R v
   $
4. 将当前秩一分量从残差中剥离
   $
     R ← R - σ u v^T
   $

重复该过程直到得到目标 rank 的前若干个奇异值。

== SVD 压缩流程

整体流程如下：

1. 读入 RGB 图像；
2. 对每个通道分别执行手写 `SVD`；
3. 保留前 $k$ 个奇异值及其对应奇异向量；
4. 重建三个通道并合成为压缩图像；
5. 计算 `PSNR` 和存储比例。

== DWT 压缩流程

`DWT` 对比方法的流程如下：

1. 对每个颜色通道做多层二维 `Haar DWT`；
2. 将所有系数按绝对值排序，保留最大的前 $N$ 个；
3. 将其余小波系数置零；
4. 执行逆变换得到重建图像；
5. 在与 `SVD` 大致相同的参数预算下，比较两种方法的重建质量。

== 评价指标

本实验采用以下两个指标：

1. 峰值信噪比 `PSNR`

设原图为 $I$，压缩后图像为 $I_"rec"$，则均方误差可写为

$
  "MSE" = 1 / (m n c) Σ (I - I_"rec")^2
$

$
  "PSNR" = 20 log_10 (255 / sqrt("MSE"))
$

`PSNR` 越大，说明图像失真越小。

2. 存储比例

存储比例定义为“压缩后参数量 / 原始像素数”。

该指标反映了压缩强度，值越小表示压缩越强。

= 实现说明

本实验的代码位于 `hw_2/option1_svd_compress.py`。整理后的代码结构主要包括以下部分：

- `manual_svd`：手写 `SVD` 主函数，负责提取前若干个奇异值；
- `precompute_image_svd` 与 `reconstruct_image_from_svd`：用于快速重建，支撑 GUI 中的实时交互；
- `haar_dwt2_once`、`haar_idwt2_once`：二维 `Haar DWT` 和逆变换；
- `compress_image_dwt`：基于系数截断的小波压缩；
- `launch_gui`：可视化展示不同 rank 下的 `SVD` 压缩效果；
- 命令行接口：支持 `svd`、`dwt` 两种模式，并支持打印对比表。

其中，GUI 版本为了避免卡顿，采用“启动时预计算一次 `SVD`，拖动滑块时只做快速重建”的方式，大幅降低了交互延迟。

= 实验环境与设置

== 实验环境

- 编程语言：Python 3.12
- 主要依赖：`numpy`、`matplotlib`、`Pillow`
- 输入图像：`CMakeTools.png`
- 测试 rank：$k in {4, 8, 16, 24, 32, 48}$

== 运行方式

SVD 压缩：

```bash
python hw_2/option1_svd_compress.py --no-gui --method svd --rank 24
```

DWT 压缩：

```bash
python hw_2/option1_svd_compress.py --no-gui --method dwt --rank 24
```

GUI 展示：

```bash
python hw_2/option1_svd_compress.py
```

= 实验结果与分析

== 原图与 GUI 展示

#figure(
  image("images/CMakeTools.png", width: 72%),
  caption: "实验使用的原始图像"
)

#figure(
  table(
    columns: 2,
    align: center,
    image("images/rank16gui.png", width: 95%),
    image("images/rank35gui.png", width: 95%),
  ),
  caption: "GUI 界面示例：左图为 rank=16，右图为 rank=35"
)

从 GUI 展示可以直观看到：随着 rank 增大，图像中的文字边缘、图标轮廓和局部色块逐渐恢复，压缩图像从较粗糙的近似逐步逼近原图。

== SVD 压缩结果

#figure(
  table(
    columns: 3,
    align: center,
    image("images/CMakeTools_svd_rank_4.png", width: 92%),
    image("images/CMakeTools_svd_rank_8.png", width: 92%),
    image("images/CMakeTools_svd_rank_16.png", width: 92%),
    ["rank = 4"], ["rank = 8"], ["rank = 16"],
    image("images/CMakeTools_svd_rank_24.png", width: 92%),
    image("images/CMakeTools_svd_rank_32.png", width: 92%),
    image("images/CMakeTools_svd_rank_48.png", width: 92%),
    ["rank = 24"], ["rank = 32"], ["rank = 48"],
  ),
  caption: "不同 rank 下的 SVD 压缩结果"
)

== DWT 压缩结果

#figure(
  table(
    columns: 3,
    align: center,
    image("images/CMakeTools_dwt_rank_4.png", width: 92%),
    image("images/CMakeTools_dwt_rank_8.png", width: 92%),
    image("images/CMakeTools_dwt_rank_16.png", width: 92%),
    ["rank = 4"], ["rank = 8"], ["rank = 16"],
    image("images/CMakeTools_dwt_rank_24.png", width: 92%),
    image("images/CMakeTools_dwt_rank_32.png", width: 92%),
    image("images/CMakeTools_dwt_rank_48.png", width: 92%),
    ["rank = 24"], ["rank = 32"], ["rank = 48"],
  ),
  caption: "不同参数预算下的 DWT 压缩结果"
)

== 定量结果

#table(
  columns: 4,
  [Rank], [SVD PSNR / dB], [DWT PSNR / dB], [Storage Ratio],
  [4],  [19.625], [20.238], [0.0128],
  [8],  [21.356], [27.957], [0.0256],
  [16], [23.852], [40.778], [0.0512],
  [24], [26.045], [58.252], [0.0769],
  [32], [28.215], [inf],    [0.1025],
  [48], [31.955], [inf],    [0.1537],
)

== 结果分析

从定量结果和图像对比可以得到以下结论：

1. `SVD` 的重建质量随 rank 增大稳定提升。`PSNR` 从 rank=4 时的 `19.625 dB` 提升到 rank=48 时的 `31.955 dB`，说明保留更多奇异值可以持续恢复图像中的主要结构和细节。

2. 在本次实验图像上，`DWT` 的表现明显优于 `SVD`。尤其在 rank=8 及以上时，`DWT` 的 `PSNR` 提升非常快，在 rank=24 时已达到 `58.252 dB`，而 `SVD` 仍为 `26.045 dB`。

3. 当 rank 达到 `32` 和 `48` 时，`DWT` 的 `PSNR` 为 `inf`，这意味着在当前实现和当前测试图像下，重建结果已经与原图完全一致或数值误差小到可以忽略。造成这一现象的主要原因是：测试图像本质上接近界面截图，包含大量平坦区域、清晰边缘和规则块状结构，这类图像对 `Haar DWT` 非常友好。

4. `SVD` 的优势在于数学解释清楚、压缩过程连续可控，并且低秩近似的含义非常直观；但对于这类具有明显局部结构和尖锐边缘的图像，简单的 `Haar DWT` 更容易以较少参数保留关键信息，因此表现更好。

5. 从运行时间体验来看，`DWT` 的命令行压缩速度也明显快于 `SVD`。这与算法复杂度相符，因为 `SVD` 需要对每个通道进行多轮矩阵迭代，而 `DWT` 的分解和重建主要由局部线性变换组成。

== 方法优缺点总结

`SVD` 的优点：

- 数学模型简洁，适合展示矩阵分解思想；
- rank 具有清晰的物理含义，便于分析压缩率与质量的关系；
- 适合作为课程作业中“矩阵分解”主题的核心方法。

`SVD` 的不足：

- 计算成本相对较高；
- 对局部边缘和高频结构的表示效率不如小波方法；
- 若图像并不具有明显低秩性，则需要更大的 rank 才能获得较好效果。

`DWT` 的优点：

- 对局部结构和边缘表达更高效；
- 算法速度快，容易得到较高 `PSNR`；
- 非常适合规则块状、界面截图类图像。

`DWT` 的不足：

- 小波基的选择会影响结果；
- 相比 `SVD`，其“矩阵分解”解释性略弱；
- 更偏向工程压缩方法，对本作业的主题支撑不如 `SVD` 直接。

= 讨论与思考

本实验中最重要的体会是：图像压缩的本质不是机械地减少像素数量，而是寻找一种更紧凑的表示方式，使主要信息可以用更少参数表达出来。`SVD` 从全局低秩结构的角度刻画冗余，`DWT` 则从频域和局部结构的角度刻画冗余，两者反映了两种不同的压缩思想。

对于本次使用的测试图像，由于其界面元素较多、边缘清晰且大面积区域颜色变化缓慢，小波压缩天然更占优势；但如果换成纹理更复杂、整体相关性更强的图像，`SVD` 与 `DWT` 的对比结论可能会发生变化。因此，本实验的结论应理解为“在当前图像和当前实现下”的实验观察，而不是对所有图像都成立的普遍规律。

= AI 辅助说明

本次实验中使用 AI 辅助完成了以下工作：

- 整理代码结构与实验报告框架；
- 将算法实现过程提炼为报告中更适合呈现的数学描述；
- 对实验现象进行文字归纳与语言润色。

所有实验结论均基于实际运行结果、生成图像和数值表进行整理，未虚构实验数据。

= 实验总结

本实验围绕 `SVD` 图像压缩完成了从数学建模、算法实现、GUI 展示到对比实验分析的完整流程。通过手写 `SVD`，我进一步理解了奇异值、奇异向量和低秩近似之间的关系；通过与 `DWT` 对比，也更直观地认识到不同压缩方法在不同图像类型上的适用性差异。

总体来看，`SVD` 作为矩阵分解方法，具有很强的理论示范意义和可解释性，十分适合作为本次作业的核心内容；而 `DWT` 则展示了更偏工程化的压缩思路，在本次实验图像上取得了更优的数值表现。两者结合起来，较完整地体现了“矩阵分解方法在多媒体压缩中的应用与比较”这一实验主题。
