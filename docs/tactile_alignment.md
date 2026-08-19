# 触觉对齐机制说明

> 仅描述动捕-触觉时间偏移 `delta_t2` 的算法。

---

## 1. 问题定义

已知：

- 动捕轨迹（BVH 关节世界坐标，帧率 \(f_m\)）
- 左右脚足底压力时间序列（帧率 \(f_p\)）
- 可选粗偏移 \(t_{\mathrm{coarse}}\)（仅作搜索中心，不作最终结果）

求标量时间偏移 \(\delta_{t2}\)，使得动捕时间轴平移后与压力时间轴对齐：

\[
t_{\mathrm{pressure}} \approx t_{\mathrm{mocap}} + \delta_{t2}
\]

当前实现提供两种估计器：

1. **旧机制**：足底贴地曲线归一化互相关
2. **新机制**：触地 / 加载起始事件匹配

---

## 2. 符号与预处理

### 2.1 坐标轴

由 `axis_preset` 决定竖直轴索引 \(v\)。默认 `zup` 下，显示坐标系中竖直轴为第 3 维（index = 2）。

### 2.2 压力侧总量

左/右脚压力：

\[
p_L(t),\quad p_R(t)
\]

若输入为 48 通道阵列，则先对通道求和得到脚总量。旧机制中再做 min-max 归一化到 \([0,1]\)：

\[
\tilde p(t)=\frac{p(t)-\min p}{\max p-\min p}
\]

不对压力总量做额外时域平滑，避免抹平着地冲击。

### 2.3 粗偏移 \(t_{\mathrm{coarse}}\)

若存在可用时钟先验，则以其为搜索中心；否则 \(t_{\mathrm{coarse}}=0\)，搜索退化为更宽的全局窗口。

---

## 3. 旧机制：足底贴地曲线互相关

### 3.1 动捕曲线

对每一帧、每一侧脚：

1. 取该侧脚相关关节集合 \(J_{\mathrm{side}}\)（优先 foot/toe/heel/ankle 类关节）。
2. 计算最低点高度

\[
h_{\mathrm{side}}(t)=\min_{j\in J_{\mathrm{side}}} z_j(t)
\]

其中 \(z_j(t)\) 为关节在竖直轴上的坐标。

3. 构造贴地程度

\[
c_{\mathrm{side}}(t)=-h_{\mathrm{side}}(t)
\]

再 min-max 到 \([0,1]\)：

\[
m_{\mathrm{side}}(t)=\mathrm{norm01}\big(c_{\mathrm{side}}(t)\big)
\]

含义：\(m\) 越大，脚越贴近地面。

### 3.2 单侧偏移搜索

对左脚与右脚分别估计。以左脚为例，候选偏移 \(\delta\) 的相关分数定义为：

1. 将动捕曲线按 \(\delta\) 插值到压力时间网格：

\[
\hat m(t)=m\big(t-\delta\big)
\]

2. 去掉无效点后中心化并除以标准差：

\[
x=\frac{\hat m-\mu_{\hat m}}{\sigma_{\hat m}},\quad
y=\frac{\tilde p-\mu_{\tilde p}}{\sigma_{\tilde p}}
\]

3. 得归一化相关

\[
S(\delta)=\mathbb{E}[x\cdot y]
\]

在搜索窗内网格搜索：

\[
\delta^\star=\arg\max_{\delta\in\mathcal{W}} S(\delta)
\]

其中默认

\[
\mathcal{W}=\big[t_{\mathrm{coarse}}-W,\ t_{\mathrm{coarse}}+W\big],\quad W=0.2\,\mathrm{s}
\]

网格步长约为压力采样周期 \(1/f_p\)。  
若无有效粗偏移先验，则 \(\mathcal{W}\) 扩展为覆盖整段信号时长的全局窗。

峰值高度：

\[
\mathrm{peak}=S(\delta^\star)
\]

### 3.3 左右融合

分别得到：

\[
\delta_L,\ \mathrm{peak}_L,\qquad \delta_R,\ \mathrm{peak}_R
\]

最终：

\[
\delta_{t2}=
\begin{cases}
(\delta_L+\delta_R)/2, & \delta_L,\delta_R\ \text{均有效}\\
\delta_L, & \text{仅左有效}\\
\delta_R, & \text{仅右有效}\\
0, & \text{均无效}
\end{cases}
\]

诊断量：左右差 \(|\delta_L-\delta_R|\)、峰值 \(\mathrm{peak}_L/\mathrm{peak}_R\)。  
差值过大或峰值过低时，结果不可信。

---

## 4. 新机制：触地事件对齐

旧机制在支撑期曲线易饱和，相关峰较钝。新机制不比较整段曲线形状，只比较**事件时刻**。

### 4.1 单位尺度

为使高度阈值与物理阈值一致，先用骨段长度估计 BVH 单位到米的尺度 \(s\)。  
例如用小腿长度中位数对齐到约 \(0.40\,\mathrm{m}\)：

\[
s=\arg\min_{s'\in\{1,0.01,0.001\}}\left|\,d\cdot s'-0.40\,\right|
\]

随后关节坐标乘以 \(s\)。

### 4.2 动捕触地事件

对单脚（以左脚为例）：

1. 取 heel / foot 与 toe 的竖直坐标最小值

\[
z(t)=\min\big(z_{\mathrm{foot}}(t),\,z_{\mathrm{toe}}(t)\big)
\]

2. 低通滤波 \(\tilde z=\mathrm{LPF}(z)\)
3. 自适应门限

\[
g = q_{2\%}(\tilde z)+0.20\cdot\big(q_{98\%}(\tilde z)-q_{2\%}(\tilde z)\big)
\]

4. 竖直速度 \(v=\mathrm{d}\tilde z/\mathrm{d}t\)
5. 在向下穿越门限且速度为负时记为触地：

\[
\tilde z(t_{i-1})\ge g > \tilde z(t_i),\quad v(t_{i-1})<0
\]

穿越时刻线性插值到亚采样精度。相邻事件最小间隔默认 \(0.35\,\mathrm{s}\)。

左右脚事件并集排序，得动捕事件集 \(\mathcal{E}_m\)。

### 4.3 压力加载起始事件

对单脚压力总量 \(p(t)\)：

1. 低通 \(\tilde p=\mathrm{LPF}(p)\)
2. 门限

\[
\tau=q_{5\%}(\tilde p)+0.25\cdot\big(q_{95\%}(\tilde p)-q_{5\%}(\tilde p)\big)
\]

3. 向上穿越 \(\tau\) 记为加载起始：

\[
\tilde p(t_{i-1})\le \tau < \tilde p(t_i)
\]

同样亚采样插值，最小间隔 \(0.35\,\mathrm{s}\)。

左右脚事件并集排序，得压力事件集 \(\mathcal{E}_p\)。

### 4.4 事件匹配与 \(\delta_{t2}\)

对每个动捕事件 \(t_m\in\mathcal{E}_m\)，在压力事件中找

\[
t_p=\arg\min_{t\in\mathcal{E}_p}\left|t-(t_m+t_{\mathrm{coarse}})\right|
\]

仅当

\[
\left|t_p-(t_m+t_{\mathrm{coarse}})\right|\le W
\]

时接受该匹配（默认 \(W=0.2\,\mathrm{s}\)），并记录

\[
d=t_p-t_m
\]

设全部接受差分为集合 \(\mathcal{D}\)：

\[
\delta_{t2}=\mathrm{median}(\mathcal{D})
\]

稳健离散度：

\[
\mathrm{scatter}=\mathrm{median}\big(|d-\delta_{t2}|\big),\quad d\in\mathcal{D}
\]

有效条件（实现中）：

- \(|\mathcal{E}_m|\ge 3\) 且 \(|\mathcal{E}_p|\ge 3\)
- \(|\mathcal{D}|\ge 3\)

否则回退为 \(t_{\mathrm{coarse}}\)，并标记不可信。

新机制左右脚事件先合并再匹配，因此输出中

\[
\delta_{t2}^{\mathrm{left}}=\delta_{t2}^{\mathrm{right}}=\delta_{t2}
\]

---

## 5. 两种机制对比

| 项目 | 旧机制 | 新机制 |
|---|---|---|
| 动捕特征 | 归一化贴地曲线 \(m(t)=-\mathrm{norm}(h(t))\) | 触地时刻集合 \(\mathcal{E}_m\) |
| 压力特征 | 归一化压力总量曲线 | 加载起始时刻集合 \(\mathcal{E}_p\) |
| 估计器 | 搜索窗内归一化互相关最大化 | 匹配事件差的中位数 |
| 搜索中心 | \(t_{\mathrm{coarse}}\) | \(t_{\mathrm{coarse}}\) |
| 默认窗口 | \(\pm 0.2\,\mathrm{s}\) | \(\pm 0.2\,\mathrm{s}\) |
| 左右信息 | 先分侧估计再平均 | 先并集事件再统一估计 |
| 主要失效模式 | 支撑期饱和导致峰钝、假峰 | 事件过少、门限失配、粗偏移偏差过大 |

---

## 6. 算法输入输出（实现对应）

### 旧机制

输入：

- `MocapFootCurveSet`: \(m_L(t), m_R(t)\)
- `PressureCurveSet`: \(\tilde p_L(t), \tilde p_R(t)\)
- `t_coarse`, `search_window_ms`

输出：

- \(\delta_{t2},\delta_L,\delta_R\)
- \(\mathrm{peak}_L,\mathrm{peak}_R\)

代码入口：`utils.pressure_alignment.estimate_pressure_alignment`

### 新机制

输入：

- BVH 关节轨迹（foot/toe 等）
- 左右脚压力总量 \(p_L,p_R\) 及其采样率
- `t_coarse`, `search_window_ms`, `axis_preset`

输出：

- \(\delta_{t2}\)
- 匹配数 \(n=|\mathcal{D}|\)
- \(\mathrm{scatter}\)
- 可信标记 `ok`

代码入口：`utils.pressure_dynamics_alignment.estimate_pressure_alignment_dynamics`  
核心函数：`align()` / `_foot_strikes()` / `_pressure_strikes()`

---

## 7. 备注

1. \(t_{\mathrm{coarse}}\) 只约束搜索，不直接作为最终偏移。  
2. 新机制的显示总曲线（触地代理 vs 压力总量）仅用于观察，不参与 \(\delta_{t2}\) 估计。  
3. 最终若人工微调，则在自动 \(\delta_{t2}\) 上叠加用户增量；这不属于自动估计算法本体。
