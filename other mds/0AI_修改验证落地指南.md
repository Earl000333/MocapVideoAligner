# AI修改验证落地指南

## 目标
让程序真正适配新的重建触觉格式，并把缺失试次在导航里正确跳过、在列表里正确显示为不可选。

## 先确认的事实
1. 新重建触觉格式以 `pressure_left.csv` / `pressure_right.csv` / `reconstruction_manifest.csv` 为主。
2. `valid_mask=1` 表示原始有效帧，`valid_mask=0` 表示重建/插值/重采样帧。
3. `missing_pressure_objects.csv` 里：
   - `C` = 触觉缺失
   - `D` = 整组数据没记录
4. 在统计和试次导航中，`C` / `D` 要直接跳过。
5. 试次列表里 `C` / `D` 仍然显示，但必须标记并禁用选择。

## 需要落地的功能
### 1. 重建触觉加载
- 新格式优先读取连续文件。
- 旧格式 `pressure_left_t*.csv` / `pressure_right_t*.csv` 继续兼容。
- 不要把单侧空 CSV 当成正常有效数据。
- 如果某个试次左右脚不完整，要能判定为缺失试次。

### 2. 触觉-动捕对齐显示
- 对应重建帧上显示 `Valid` 标记。
- 曲线中：
  - `valid_mask=1` 用虚线
  - 其他帧用实线

### 3. 试次跳过规则
- 读取 `missing_pressure_objects.csv`。
- `C` / `D` 试次在“下一试次 / 上一试次 / 选择批次”中都要被排除。
- 选择批次时不要删除这些项，要保留显示，但置灰/不可选。
- 若某个实际重建目录左右脚一侧为空，也应视为 `C`。

## 修改范围
优先检查并修改这些文件：
- `utils/pressure_alignment.py`
- `ui/pressure_alignment_page.py`
- `ui/main_window.py`
- `tests/test_pressure_alignment.py`
- `config.py`

## 推荐实施顺序
1. 先用 `git status` 和 `git diff` 看当前工作区是否真的有改动。
2. 在 `utils/pressure_alignment.py` 里完成新格式读取、`valid_mask` 保留、缺失判定。
3. 在 `ui/pressure_alignment_page.py` 里完成 `Valid` 标记和虚实线绘制。
4. 在 `ui/main_window.py` 里接入 `missing_pressure_objects.csv`，实现跳过和禁用选择。
5. 补测试，覆盖：
   - 新连续格式
   - 旧分段格式
   - 单侧空 CSV
   - `C` / `D` 跳过
6. 运行测试并确认真实样例能被正确跳过。

## 验证标准
必须满足：
- `S1301_2` 这类实际右侧为空的重建试次不能再被当成正常试次加载。
- “下一试次”不会停在 `C` / `D`。
- “选择批次”里 `C` / `D` 仍可见，但不可选。
- 重建触觉帧上能看到 `Valid`。
- 曲线虚实线规则正确。

## 额外要求
- 不要只回复“已修改”，必须给出实际文件差异和验证结果。
- 如果发现之前的改动没真正落盘，先纠正到当前工作区，再说完成。
- 如果程序仍显示旧状态，优先怀疑：
  - 没重启程序
  - 启动的不是当前工作区代码
  - 修改没有写入正确文件

