# TinyZero 实验历史、指标与结果分析

## 1. 口径声明

本文件严格区分三类指标：

1. **规则平均奖励**：正确表达式 1 分，格式/数字合法但结果错误 0.1 分，无法解析 0 分。它不是准确率。
2. **严格正确率（exact）**：最终表达式必须使用全部给定数字恰好一次，且计算值等于 target。
3. **格式率（format）**：能按协议输出并通过相应格式检查；不能替代数学正确率。

当前训练使用的 `test.parquet` 在训练中被反复评估并用于选择 checkpoint，因此本文称其为“训练内验证集”。虽然文件名叫 test，但不具备最终盲测属性。

本地工作区未保存实验日志/checkpoint。下列 B 级结果来自此前会话中已经读取的服务器日志与严格评测输出；历史日志位于授权服务器的 `<PROJECT_ROOT>/logs/countdown-step300-continue-200.log`，checkpoint 位于服务器实验目录。2026-11-12 本轮只读 SSH 复查超时，交接前必须归档这些原始文件。

## 2. 实验汇总表

| 实验版本 | 目的/改动 | 数据规模 | 核心参数 | 评测指标 | 结果 | 相对变化 | 证据路径/等级 |
|---|---|---:|---|---|---|---|---|
| E0 官方默认方案 | 提供可运行 PPO baseline 配置 | 默认 train 327,680；val 1,024 | actor lr 1e-6；critic lr 1e-5；KL 0.001；entropy 0.001；batch 256；15 epochs | 无本地结果 | 只有配置，不能作为实验结果 | N/A | `scripts/train_tiny_zero.sh`，A/D |
| E1 初始 PPO 训练 | 从基础模型做 Countdown PPO，选取 step 300 checkpoint | 训练集/验证集路径曾核验；具体有效行数应从日志复核 | 沿用 TinyZero PPO 主配置；完整 resolved config 当前未归档 | 严格 exact；mean reward | step 300：exact 51.27%；mean reward 0.55918 | 缺基础模型同口径 strict baseline，无法算相对基础模型提升 | 历史严格评测输出，B；远端原始文件待归档 |
| E2 续训前 baseline validation | 记录恢复 checkpoint 在训练内验证集上的起点 | validation 1,024 | step 300 checkpoint，未更新 | val mean reward | 0.5564453 | 作为 E3 内部基线 | 续训日志，B |
| E3 稳定性续训 | lr 降至 2e-7；KL 提至 0.005；entropy 0.001；每 50 step 验证/保存；约 200 updates | 同 E1；validation 1,024 | actor lr 2e-7；KL 0.005；entropy 0.001；test/save 50 | val mean reward | step50 0.559；100 0.561；150 0.568；200 0.567；final step201 0.566699 | 相对续训前 +0.011555（step150，约 +2.08%）；末段略回落 | `<PROJECT_ROOT>/logs/countdown-step300-continue-200.log`，B |
| E4 原始 checkpoint 严格评测 | 建立同口径参照 | 1,024 | step 300；严格规则 | exact / mean reward | 51.27% / 0.55918 | 参照 | 历史严格评测输出，B |
| E5 最优续训 checkpoint 严格评测 | 验证 val 最优点是否转化为 exact 改善 | 1,024 | continuation `global_step_150` | exact / format / mean reward / 分组 exact | 535/1024=52.246%；format 99.219%；mean reward 0.569434；3-number 74.49%；4-number 31.835% | 对 E4：exact +0.976 pp（约 +1.90% 相对）；reward +0.010254（约 +1.83% 相对） | 历史严格评测输出，B |
| E6 单次梯度尖峰审计 | 判断 step160 是否发散 | 单个训练 step 及邻域 | grad clip 1.0 | actor grad norm、KL、entropy、后续 val | step160=8.693；155–159 约0.518–0.930；161–165 约0.517–0.856；KL≈0.029，entropy≈0.007；后续 val 稳定 | 单点异常，无持续恶化 | 续训日志 + `dp_actor.py:143-151`，A/B |
| E7 400 条冷启动 SFT | 计划用最优模型多样采样、规则过滤，再对 base model SFT | 目标 400 条 | 未执行 | 未定义 | 无结果 | N/A | 仅方案讨论，E |

## 3. Baseline 定义问题

当前有两个容易混淆的 baseline：

- E2 是**续训 baseline**：已经做过 PPO 的 step 300 checkpoint；它不是基础模型。
- E4 是**旧 checkpoint 对照**：与 E5 做严格评测的同口径参照；仍不是未训练基础模型。

真正回答“RL 相比基础模型提升多少”需要对原始 base model 使用同一 1,024 条或新的 held-out set、相同 prompt、相同 sampling/decoding 和严格 scorer 评测。当前证据缺失，所以不得写“从 X 提升至 52.246%”。可写的唯一严谨结论是：

> 在同一评测口径下，稳定性续训后的 step 150 checkpoint 相比原始 PPO step 300 checkpoint，严格正确率提高 0.976 个百分点，规则平均奖励提高 0.010254。

## 4. 初始 PPO 实验

### 4.1 假设

基础模型在 Countdown 上已经具备一定算术和格式生成能力，规则 outcome reward 配合 PPO 能增加正确表达式概率，并逐步形成搜索/反思文本模式。

### 4.2 设置

代码路径显示默认方案使用 GAE/PPO、单 response rollout、独立 critic、固定 reference policy 和固定 KL controller。训练脚本的 actor lr 为 `1e-6`、critic lr `1e-5`、KL `0.001`、max response `1024`、train batch `256`、mini/micro batch `64/8`。

但由于初始实验的 resolved Hydra config 当前不在本地，不能保证所有参数与脚本当前版本完全一致。尤其模型路径、GPU 数、rollout tensor parallel、总 training steps、恢复点和随机种子必须以原日志为准。

### 4.3 结果

历史严格评测记录：step 300 checkpoint exact `51.27%`、规则 mean reward `0.55918`。没有基础模型 strict baseline，因此只能把它作为后续续训的对照。

## 5. 稳定性续训实验

### 5.1 问题与假设

初始训练后担心两类风险：学习率较大导致 checkpoint 附近震荡或策略快速漂移；KL 约束过弱/entropy 过低可能引起生成分布退化。续训采用保守更新：

- actor lr `1e-6 → 2e-7`（降至 20%）；
- KL coefficient `0.001 → 0.005`（5 倍）；
- entropy coefficient 明确设为 `0.001`；
- 每 50 step 验证并保存，便于及时早停。

假设是：更小步长、更强 reference 约束和适度探索可在保留既有能力的同时获得小幅稳定提升。

### 5.2 验证轨迹

| continuation step | val mean reward | 相对 step0 绝对变化 | 解释 |
|---:|---:|---:|---|
| 0 | 0.5564453 | 0 | 恢复点 baseline |
| 50 | 0.559 | +0.0025547 | 小幅提升 |
| 100 | 0.561 | +0.0045547 | 继续提升 |
| 150 | 0.568 | +0.0115547 | 日志内最优点 |
| 200 | 0.567 | +0.0105547 | 略低于最优 |
| 201 final | 0.566699 | +0.0102537 | final validation，仍高于 baseline |

step 150 到 step 201 下降 `0.001301`，幅度很小，但说明“最新 checkpoint”不等于“最好 checkpoint”。选择 `global_step_150` 做严格评测是合理的；若继续训练，应使用独立验证策略并设置 patience，而不是依据最后一次训练 loss。

### 5.3 严格评测

step 150 的严格结果：

- exact：`535/1024 = 52.246%`；
- format：`99.219%`，约 1,016/1,024（由百分比推回的整数仅作近似，原始逐样本结果待归档）；
- mean rule reward：`0.569434`；
- 3-number exact：`74.49%`；
- 4-number exact：`31.835%`。

3 数和 4 数题的准确率差距约 `42.655` 个百分点，说明难度随操作数组合空间显著上升。整体改进可能来自某类题的局部提升；当前缺少旧 checkpoint 的同类分组结果，因此不能判断续训主要改善了 3 数还是 4 数问题。

### 5.4 改进可信度

535/1024 的二项比例标准误约为：

\[
SE=\sqrt{p(1-p)/n}\approx 1.56\%.
\]

单模型 95% 区间的半宽约 3.1 个百分点。E4→E5 的 +0.976 pp 小于这一尺度，而且两个 checkpoint 在同一题集上，应使用逐题配对结果做 McNemar test 或 paired bootstrap；当前只有汇总值，无法验证统计显著性。因此结论应是“观察到小幅提升”，而不是“已证明显著提升”。

## 6. 梯度尖峰、熵坍缩与 KL 审计

step 160 的 `actor/grad_norm=8.693` 是曲线中的孤立尖峰。相邻值：

| step | grad norm |
|---:|---:|
| 155 | 0.633 |
| 156 | 0.727 |
| 157 | 0.518 |
| 158 | 0.930 |
| 159 | 0.627 |
| 160 | 8.693 |
| 161 | 0.856 |
| 162 | 0.606 |
| 163 | 0.744 |
| 164 | 0.517 |
| 165 | 0.668 |

代码记录的是 `clip_grad_norm_` 返回值，即裁剪前范数；配置上限为 1.0。尖峰后一轮立即恢复，同期 KL 约 0.029、entropy 约 0.007，验证奖励没有断崖下降。因此：

- 没有证据表明发生持续梯度爆炸；
- 没有证据表明发生 entropy collapse；
- 没有证据表明 KL 持续失控；
- 最合理解释是单个高方差/outlier batch，被梯度裁剪吸收。

但 entropy 的绝对值受 token 分布和实现口径影响，仅凭 `0.007` 不能判断“健康”；应观察时间趋势、response diversity、重复率和 greedy/sample performance。

## 7. 自动评测机制

`RayPPOTrainer.fit()` 在 `global_steps % test_freq == 0` 时调用 `_validate()`，训练达到总步数后再次 final validation；所以续训配置 `test_freq=50` 会自动在 50、100、150、200 等点评测。

自动验证使用与训练相同的规则 reward function，但这不等同于完整严格评测报告。训练曲线最应关注：

- `val/test_score/countdown` 或对应 validation reward；
- strict exact accuracy（外部/扩展 scorer）；
- format accuracy；
- 3-number / 4-number 分组准确率；
- actor entropy、KL、pg_clipfrac、grad_norm；
- response length 和吞吐/step time。

其中训练 reward 可能被 0.1 的格式分放大，最终模型选择应优先 exact accuracy；若只能在线记录 reward，至少同时记录各奖励桶 `{0,0.1,1}` 的占比。

## 8. 失败、负面结果与缺失实验

### 已有证据的工程失败

1. **网络/代理问题**：GitHub clone、conda/pip 曾因 307、连接超时、错误 SOCKS 端口失败；最终通过本地下载/上传或可用网络路径解决。它属于环境交付问题，不是算法实验。
2. **pip virtualenv policy**：`PIP_REQUIRE_VIRTUALENV` 导致已激活 conda 环境仍拒绝安装；通过清理该环境变量处理。
3. **W&B 登录误用**：曾输入非 40 字符 key；之后完成登录。聊天中出现过 API key，应立即轮换，报告不保留任何密钥。
4. **依赖冲突**：安装 `wandb==0.18.7` 把 protobuf 降到 5.29.6，与 SwanLab 0.10.0 所需 protobuf `>=6.31.1,<7` 冲突。最终实验选择 W&B，SwanLab 不应声明已用于训练可视化。
5. **过长训练预估**：初始 19,199 step、约 48 秒/step 的估算达到约 256 小时，促使改为受控的短续训/更频繁验证。

### 没有完成或没有结果的算法实验

- base model strict baseline；
- PPO vs GRPO 同口径、同生成预算对照；
- actor lr `2e-7` vs `5e-7` 消融；
- KL `0.001` vs `0.005` 单变量消融；
- entropy coefficient 消融；
- 多随机种子；
- 400 条冷启动 SFT；
- held-out test；
- 人工审计思维过程忠实性。

因此 E3 同时改了三个主要超参数，提升不能归因到其中任何一个单独因素。

## 9. 结果冲突与解释

续训 step 150 的在线 validation reward `0.568`，严格评测 mean reward `0.569434`，数值接近但不完全一致。可能原因包括采样随机性、评测 temperature、批次/seed、checkpoint 加载、在线与离线 scorer 细节。两者不能强行视为同一次评测。

初始 step 300 strict mean reward `0.55918` 与续训前在线 baseline `0.5564453` 也不同，进一步说明评测存在采样方差或配置差异。交接前必须保存：评测命令、随机种子、temperature/top-p、生成次数、模型 checkpoint 哈希和逐样本输出。

## 10. 最终实验结论

可证实的最重要结果是：保守续训在训练内验证集上呈现连续提升并在 step 150 达峰；同口径严格评测相对旧 PPO checkpoint 提高约 0.976 个百分点 exact accuracy。提升为正且与 mean reward 同向，但幅度较小，缺少配对显著性检验和独立盲测，不能过度外推。

当前最合理的决策是停止继续无目标续训，先补齐 base baseline、逐样本配对结果和 held-out test；随后再决定进行冷启动 SFT、GRPO 对照或更长训练。
