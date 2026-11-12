# 个人贡献、技术亮点与经验沉淀

## 1. 归属原则

本仓库 `git shortlog -sne --all` 未出现 `zihao.hu4`，当前工作树也没有未提交的代码改动。因此不能依据 Git 将 TinyZero/veRL 的数据脚本、PPO 实现、Ray/FSDP/vLLM 框架或 reward function 归为个人原创。

本报告将贡献分为：

- **已核实的实验与工程工作**：此前会话中有命令、日志、checkpoint 或结果证据；
- **上游项目能力**：代码真实存在，但作者是上游贡献者；
- **待确认个人贡献**：可能参与，但缺 commit、文档签名或产物元数据，不能直接认领。

## 2. 已核实的个人工作

### 2.1 训练环境与资产部署

完成了训练服务器连接、conda 环境定位与激活、模型和数据从本地到服务器的部署，并处理企业网络/代理、pip virtualenv policy、W&B/SwanLab 依赖冲突等问题。形成的工程经验包括：

- conda 环境用完整路径激活时，目标目录必须已经是合法 conda env；
- `PIP_REQUIRE_VIRTUALENV` 会让 conda 环境中的 pip 仍拒绝安装，需要明确检查并按环境策略处理；
- 服务器本地没有 SOCKS 监听时，设置 `ALL_PROXY=socks5h://127.0.0.1:10808` 只会造成 connection refused；
- W&B 与 SwanLab 对 protobuf 的版本约束可能冲突，应固定一套 tracker 依赖并导出 lock/conda spec；
- API key 不应出现在聊天、日志或 shell history，已经暴露的 key 必须轮换。

### 2.2 初始 PPO 训练链路跑通

基于现有 TinyZero/veRL 代码，完成 Countdown PPO 训练启动、进度观察、checkpoint 保存频率与自动验证确认，并结合 W&B 曲线监控 reward、entropy、KL、gradient norm 等指标。

贡献性质是“复现与实验工程”，不是声称原创 PPO 框架。

### 2.3 训练时长与资源问题分析

原始日志显示约 `19,199` steps、早期约 `48 s/step`，简单外推超过 10 天。及时识别“总 epoch/数据规模/每步 rollout 成本”共同导致训练周期过长，并转向 checkpoint 恢复、短程续训和高频验证，而不是让作业无限运行。

### 2.4 稳定性续训方案设计与执行

针对策略漂移、KL 与 entropy 风险，设计并执行：

- actor lr：`1e-6 → 2e-7`；
- KL coefficient：`0.001 → 0.005`；
- entropy coefficient：`0.001`；
- 额外约 200 step；
- `test_freq=50`、`save_freq=50`。

结果是训练内 validation reward 从 `0.5564453` 上升至 step 150 的 `0.568`，随后略回落。该实验验证“保守续训不会立即破坏既有能力，并观察到小幅提升”，但由于三个参数同时改变，不能把增益归因到单个超参数。

### 2.5 异常曲线诊断

对 step 160 的 `actor/grad_norm=8.693` 进行了代码与日志联合判断：

- 对照相邻 step 确认是单点尖峰；
- 定位 `clip_grad_norm_(max_norm=1.0)`，确认图中是裁剪前范数；
- 检查同期 KL、entropy 和后续 validation 未恶化；
- 排除“仅凭一个尖峰就判定训练发散/熵坍缩”的错误结论。

这体现了从指标到实现语义的排障方法：先确认指标定义，再看邻域和关联指标，最后判断是否影响最终性能。

### 2.6 严格评测与 checkpoint 选择

没有盲目使用最后 checkpoint，而是依据 validation 选择 step 150，并对旧 step 300 与新 step 150 进行同口径严格评测。新模型获得 exact `52.246%`、format `99.219%`、mean reward `0.569434`，相对旧模型 exact `51.27%` 提升 `0.976` 个百分点。

同时识别 3-number `74.49%`、4-number `31.835%` 的显著难度差异，说明整体平均分掩盖了题型结构。

### 2.7 数据泄漏风险识别

在规划用最佳模型生成 400 条正确推理数据进行冷启动 SFT 时，明确指出：不能从当前 test/validation 集生成训练样本；题目应按 `(sorted(numbers), target)` 规范化去重；现有 test 已用于 checkpoint 选择，应降级为 validation，并新建 held-out test。

该工作目前停留在方案和风险控制，尚未生成数据或训练 SFT 模型。

## 3. 待确认的个人贡献

以下事项没有 Git/文件元数据支持，需要人工补充：

- 是否修改过远端训练脚本、进度条或恢复训练逻辑，但没有同步回当前仓库；
- 严格评测脚本是否由本人编写，文件位置和版本是什么；
- 模型/数据上传脚本、服务器启动命令和 W&B run owner；
- 是否有内部文档、周报、实验表或 PR 能证明方案设计归属；
- checkpoint 目录和日志文件的创建者/时间戳；
- 是否完成过 GPU keep-alive 矩阵脚本。若只是占卡且不服务训练，不应作为技术成果，且可能违反资源政策。

交接前应以 commit、文件、任务记录或实验链接补齐，不能仅凭回忆认领。

## 4. 五个技术亮点

### 亮点一：可验证奖励替代神经奖励模型

**原始问题**：推理质量通常难以低成本标注，神经 reward model 还可能被利用。  
**技术方案**：把任务限制为 Countdown，最终表达式可程序验证。  
**关键实现**：`verl/utils/reward_score/countdown.py` 提取答案、检查数字多重集、受限求值，返回 0/0.1/1。  
**实验证据**：初始与续训均可直接得到 validation reward 和 strict exact。  
**最终效果**：无需额外 reward model 即建立 RL 闭环，但只能保证最终答案，不保证中间推理忠实。

### 亮点二：veRL 混合引擎完成高效在线 RL

**原始问题**：LLM PPO 同时需要 actor、reference、critic 和高吞吐生成，显存和调度复杂。  
**技术方案**：Ray 编排 + FSDP 训练 + vLLM rollout + actor/rollout 权重共享。  
**关键实现**：`main_ppo.py`、`ray_trainer.py`、`fsdp_workers.py`。  
**实验证据**：训练和 checkpoint 恢复链路实际跑通。  
**最终效果**：在有限集群资源上完成 PPO；但资源利用率、成本和吞吐没有系统归档。

### 亮点三：稳定性导向的续训策略

**原始问题**：已有 checkpoint 继续用较高 lr 可能造成策略漂移，低 entropy/KL 失控可能破坏生成质量。  
**技术方案**：lr 降至 2e-7、KL 提至 0.005、entropy bonus 0.001，并每 50 step 验证/保存。  
**关键实现**：Hydra 参数覆盖与 checkpoint 恢复。  
**实验证据**：val reward 0.5564453→0.568；strict exact 51.27%→52.246%。  
**最终效果**：观察到小幅正向提升并定位 step150 最优；尚无消融和独立盲测。

### 亮点四：从曲线语义而非表面数值诊断训练

**原始问题**：单次 grad norm 激增容易被误判为梯度爆炸。  
**技术方案**：联合检查代码中的梯度裁剪返回语义、前后 step、KL/entropy 和验证曲线。  
**关键实现**：`dp_actor.py:143-151`。  
**实验证据**：step160 8.693 后立刻恢复，验证无崩溃。  
**最终效果**：避免不必要中止或错误调参，形成可复用监控方法。

### 亮点五：把训练内验证与最终泛化评测分开

**原始问题**：文件名 `test.parquet` 容易让人误认为是盲测集，但训练过程中反复使用。  
**技术方案**：将其重新定义为 validation，规划规范化去重的 generation pool 与新 held-out test。  
**关键实现**：审计 `countdown.py` 连续切分和 `train_tiny_zero.sh` 的 val_files。  
**实验证据**：当前严格结果确实基于 1,024 条已参与 checkpoint 选择的数据。  
**最终效果**：避免把验证增益夸大为泛化结论；独立测试仍待完成。

## 5. 关键经验

### 5.1 指标必须先定义再比较

mean reward 包含 0.1 格式分，不等于 exact accuracy；在线 validation 与离线 strict eval 采样配置不同也不能直接比较。任何实验表都应记录 scorer 版本、decoding 参数、seed 和逐样本输出。

### 5.2 最后 checkpoint 不等于最佳 checkpoint

step150 比 step200/201 略好。训练应高频验证、保留 top-k checkpoint，并使用 patience early stopping。

### 5.3 多变量调参只能证明组合有效

同时改变 lr、KL、entropy 得到提升，只能说明组合方案可行。若要形成算法结论，需单变量或正交消融。

### 5.4 训练曲线不能替代最终任务指标

pg loss、grad norm、KL、entropy 用于诊断稳定性，最终决策仍应看 exact、分组准确率和独立 held-out 泛化。

### 5.5 复现资产和代码同等重要

没有 resolved config、日志、checkpoint manifest、评测脚本和逐样本结果，几个月后就无法严格复盘。每次实验应自动写入：Git SHA、命令、环境 freeze、GPU 信息、数据 hash、seed、checkpoint hash 与 metrics JSON。

## 6. 当前不足

- 没有 base model 同口径 baseline，无法量化完整 RL 收益。
- 没有 held-out test，现有 52.246% 可能受 checkpoint selection bias 影响。
- 没有多 seed 与统计显著性检验，0.976 pp 提升可能落在采样波动范围。
- 3/4-number 分组只对最终模型有记录，缺少版本间分组对照。
- 思维过程只做格式约束，没有过程正确性或忠实性评测。
- 严格评测和续训脚本没有进入当前 Git 工作树。
- Git 无个人身份记录，不利于贡献审计。

## 7. 建议的个人成果表述

可核实、不过度声明的表述：

> 基于 TinyZero/veRL 搭建 Qwen Countdown PPO 训练与 W&B 监控链路，完成模型/数据部署、checkpoint 恢复和严格规则评测；针对训练稳定性将 actor 学习率由 1e-6 降至 2e-7、KL 系数由 0.001 提至 0.005，并按 50 step 验证/保存。续训最优 checkpoint 在 1,024 条验证题上的严格正确率由 51.27% 提升至 52.246%（+0.976 pp）；同时识别测试集复用、奖励/准确率口径差异和单次梯度尖峰等风险，提出独立 held-out test 与规范化去重方案。

不应表述为“独立实现 PPO/veRL/TinyZero 框架”，除非后续能提供本人提交或独立代码证据。
