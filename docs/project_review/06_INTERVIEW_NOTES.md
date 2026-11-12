# TinyZero 技术面试说明与问答

## 1. 三分钟项目介绍

我参与的是一个基于 TinyZero/veRL 的可验证数学推理强化学习复现实验。任务是 Countdown：模型拿到 3–4 个数字和一个目标值，需要使用每个数字恰好一次，通过四则运算得到目标，并以 `<think>` 和 `<answer>` 格式输出。这个任务的优势是最终答案可以用程序规则精确验证，不需要训练 reward model。

技术链路上，数据由 Hugging Face Countdown 数据集转换成 Parquet，里面包含 prompt、target、numbers 和规则奖励元数据。训练使用 veRL：Ray 负责任务编排，vLLM 负责在线 rollout，PyTorch FSDP 负责 actor、critic、reference 的分布式训练。实际算法是 PPO+GAE。模型生成 response 后，规则 scorer 检查答案格式、数字是否恰好使用一次、表达式是否等于目标；终点奖励再减去相对 reference policy 的 token-level KL 惩罚，critic 估计 value，GAE 计算 advantage，最后分别更新 critic 和 actor。

我的工作重点不是声称重写了 veRL，而是完成环境和模型/数据部署、跑通 PPO 训练、W&B 监控、checkpoint 恢复、稳定性调参与严格评测。针对已有 checkpoint 继续训练时可能发生策略漂移的问题，我把 actor 学习率从 `1e-6` 降到 `2e-7`，KL 系数从 `0.001` 提高到 `0.005`，entropy bonus 设为 `0.001`，并把验证和保存频率缩短到 50 step。训练内 validation reward 从 `0.5564453` 提升到 step150 的 `0.568`，之后略回落，所以选择 step150 而不是最后 checkpoint。

严格评测上，旧 step300 checkpoint 的 exact 是 `51.27%`、mean reward `0.55918`；续训 step150 是 `535/1024=52.246%`、mean reward `0.569434`、format `99.219%`，exact 提升 `0.976` 个百分点。3-number 是 `74.49%`，4-number 只有 `31.835%`，说明任务难度结构差异明显。

我也识别了结论边界：当前所谓 test 文件在训练中被反复用于选 checkpoint，实际是 validation；缺少独立 held-out test、基础模型同口径 baseline、多随机种子和配对显著性检验。因此这个结果只能表述为对旧 PPO checkpoint 的小幅观察性提升，不能宣称已经证明泛化显著提高。下一步应先补齐盲测和实验资产，再考虑用正确样本做 400 条冷启动 SFT 或进行 PPO/GRPO 对照。

## 2. 三个最重要的技术难点

### 难点一：稀疏终点奖励如何稳定训练长序列

Countdown 只有最终表达式能被精确打分，前面的推理 token 没有过程标签。项目通过 critic 和 GAE 把终点奖励传播到整段响应，并用 PPO clip 限制单次策略更新；KL penalty 保持策略接近 reference，entropy bonus 保持探索。工程上还需要长达 1,024 token 的 rollout，使每步成本和方差都很高。

### 难点二：LLM PPO 的显存和调度

同一轮需要 actor、old policy log-prob、reference log-prob、critic value 和 rollout engine。项目使用 FSDP 分片、vLLM 生成、Ray worker group、remove padding 和 dynamic batch。Git 历史显示 dynamic batch/remove padding 和 mini-batch 64 是为修复 reference log-prob 阶段 Ray OOM。

### 难点三：如何判断训练真的变好

训练 reward 包含格式分，不能当 exact accuracy；文件名 test 实际是 validation；最后 checkpoint 不一定最好；单次梯度尖峰也不一定发散。需要同时看 strict exact、format、分题型结果、KL、entropy、clip fraction、grad norm，并建立独立 held-out test 和逐题配对统计。

## 3. 核心算法速记

### 3.1 规则奖励

\[
r\in\{0,0.1,1\}
\]

`1`：数字使用严格正确且表达式等于 target；`0.1`：数字/格式基本合法但结果错误；`0`：无法提取答案。奖励只放在最后有效 response token。

### 3.2 KL-shaped reward

\[
r_t'=r_t-\beta(\log\pi_{old}(a_t|s_t)-\log\pi_{ref}(a_t|s_t))
\]

reference 防止策略为了 reward 过度偏离预训练语言分布。

### 3.3 GAE

\[
\delta_t=r_t'+\gamma V_{t+1}-V_t,
\quad A_t=\delta_t+\gamma\lambda A_{t+1}
\]

代码默认 `γ=λ=1`，并对有效 token 的 advantage whiten。

### 3.4 PPO clipped objective

\[
L=-E[\min(\rho A,clip(\rho,1-\epsilon,1+\epsilon)A)]-c_H H
\]

`ε=0.2`。熵项鼓励探索，梯度范数裁剪上限 1.0。

### 3.5 GRPO

同一 prompt 生成多条回答，在组内用 reward 均值/标准差归一化：

\[
A_i=(R_i-\mu_g)/(\sigma_g+\epsilon)
\]

可省 critic，但增加 rollout，且组内 reward 全相同时信号弱。当前代码支持，实际实验没有运行。

## 4. 为什么选择该技术方案

- Countdown 可程序验证，适合低成本探索 RL reasoning。
- PPO 训练链路在当时的 veRL 中成熟，actor-critic、KL、checkpoint 和指标完整。
- vLLM 是生成瓶颈的合理选择；FSDP 降低多模型显存压力；Ray 适合多角色 worker 编排。
- rule reward 比神经 reward model 更可解释，但无法验证 CoT 忠实性。
- 续训采用低 lr/高 KL/entropy 是稳定性优先的保守策略，适合从已有 checkpoint 小步改进。

## 5. 真实指标及正确解释

| 指标 | 旧 PPO step300 | 续训 step150 | 正确解读 |
|---|---:|---:|---|
| strict exact | 51.27% | 52.246% | +0.976 pp；同一评测口径，小幅观察性提升 |
| mean rule reward | 0.55918 | 0.569434 | +0.010254；包含格式分 |
| format | 未归档 | 99.219% | 只说明输出协议好，不代表答案正确 |
| 3-number exact | 未归档 | 74.49% | 较容易 |
| 4-number exact | 未归档 | 31.835% | 主要瓶颈 |

不能说“比 base model 提升 0.976 pp”，因为旧模型已经是 PPO checkpoint；不能说“显著提升”，因为缺逐题配对检验/多 seed；不能说“测试集泛化”，因为该集合参与选 checkpoint。

## 6. 十五个追问与参考答案

### Q1：为什么不用 SFT，直接 RL？

项目目标是复现 R1-Zero 风格现象，验证基础模型能否依靠可验证奖励发展策略，而不是模仿人工 CoT。直接 RL 也省去高质量推理标注。但它训练不稳定、样本效率低，所以冷启动 SFT 是后续可比较方案，而不是原实验前提。

### Q2：为什么用 PPO 而不是 GRPO？

当时官方脚本和已保存 checkpoint 是 PPO+GAE 路径，actor/critic/ref 和恢复机制已跑通。GRPO 在代码中可用但需要每题多采样和新的预算设计。没有做同预算对照，所以不能声称 PPO 更优；如果重做，我会用新 veRL 比较 wall-clock、生成 token 预算和 exact。

### Q3：规则奖励是否会被 reward hacking？

会有风险。当前用数字多重集校验和字符白名单降低攻击面，但只验证最终表达式，不能保证推理文本正确。解析器边界也可能被利用。应改用 AST 解释器、保存错误桶、做 adversarial test，并抽检 CoT 忠实性。

### Q4：为什么错误答案还给 0.1？

它提供格式 shaping，让早期几乎不会输出协议的模型先学会 `<answer>` 和数字使用规则。但它会让 mean reward 与 exact 脱钩。更严谨的做法是分别记录 format reward 和 correctness reward，最终选模以 exact 为主。

### Q5：学习率为什么从 1e-6 降到 2e-7？

从已经具备能力的 checkpoint 继续训练，目标是局部改进而不是重塑策略。较小 lr 降低单 batch 高方差造成灾难性更新的风险。实际 val 曲线稳步上升到 step150，但没有 5e-7 对照，所以只能说组合方案可行。

### Q6：KL coefficient 提高 5 倍有什么作用？

KL penalty 抑制 actor 偏离 reference。提高 β 会牺牲部分探索换取稳定性和语言分布保持。同期没有 KL 持续爆炸，最终性能小幅提高；但由于 lr/entropy 同时变化，不能单独归因给 KL。

### Q7：entropy coefficient 是什么？

actor loss 中减去 `c_H * entropy`，最小化 loss 就会鼓励更高熵，避免分布过早塌缩到少数表达式。过大则答案随机，过小则探索不足。当前为 0.001，需要结合 entropy 趋势和输出 diversity，而不是只看一个绝对值。

### Q8：step160 grad norm 8.693 为什么没停？

代码的 `clip_grad_norm_` 返回裁剪前 norm，实际更新被裁到 1.0。尖峰只有一个 step，前后回到 0.5–0.9，KL/entropy/validation 都稳定。因此更像 outlier batch，不是持续梯度爆炸。

### Q9：为什么每 50 step 验证？

续训只有约 200 step，100-step 间隔只能看到很少点。50-step 可以识别 step150 最优并避免最后一步轻微退化。但验证太频繁增加时间并强化 selection bias，所以必须另设 held-out test。

### Q10：为什么 final checkpoint 不是最优？

RL 指标高方差，优化目标和验证 exact 不完全一致；继续更新可能离开局部最佳。step150 val 0.568，step200/201 约0.567/0.566699，因此按验证选 step150。

### Q11：0.976 pp 提升可信吗？

方向上 old/new 的 strict exact 和 mean reward 都同向，但幅度小。1024 条的单比例标准误约1.56%，需要同题配对结果做 McNemar/paired bootstrap，并用独立 test、多 seed 才能判断可靠性。

### Q12：为什么 4-number 准确率低这么多？

4 个操作数的排列、括号和操作符组合空间显著更大，搜索深度更高，中间分数/除法也更复杂。整体瓶颈主要在 4-number，但要确认改进来源，还需 old checkpoint 的分组结果和错误类型分析。

### Q13：400 条正确推理数据能否用于冷启动？

可以做 proof-of-concept，但要从 generation train pool 多样采样、规则过滤、去重和人工抽检，不能用 test 题。400 条容易过拟合，应比较 base、SFT、PPO、SFT+RL，并保留独立 held-out test。

### Q14：当前最大评测问题是什么？

`test.parquet` 在训练中被反复用于选 checkpoint，已经是 validation。缺少从未查看的 held-out test，所以不能把52.246%当作无偏泛化结果。

### Q15：如果重新做，你会改变什么？

先建立数据 canonical split、base baseline、逐样本评测和实验 manifest；迁移到新 veRL；在同生成预算下比较 PPO/GRPO；将 reward 分量化；做 3 seed 和配对统计；最后再做冷启动 SFT。这样先保证证据质量，再追求指标。

## 7. 进一步可能追问

### PPO clip 与 KL 是否重复？

不完全重复。PPO clip 限制相对 rollout old policy 的单次更新；KL penalty 约束当前/old action distribution 相对固定 reference 的长期漂移。前者是局部 trust-region 近似，后者是保持预训练先验。

### 为什么需要 critic？

critic 提供状态相关 baseline，降低 policy gradient 方差，并把终点奖励传播到不同 token。但 critic 本身带来显存/计算和估计误差，这正是 GRPO 去 critic 的动机之一。

### 如何衡量思维链质量？

当前 scorer 不衡量。可以解析每一步算式、检查中间等式和可用数字状态；或人工标注一小批“过程正确/与答案一致/是否自我纠错”，并报告与 final exact 的相关性。不能仅凭 `<think>` 长度或出现反思词判断。

### 如何避免数据泄漏？

先对题目建立 canonical key `(tuple(sorted(nums)), target)`，再按 key 分 train/generation/validation/held-out；任何用于生成 SFT、调参、选 checkpoint 的题都不能进入最终 test。

## 8. 面试中的保密边界

- 可以介绍开源 TinyZero/veRL 架构和公开算法。
- 只报告经允许的聚合指标，不展示内部服务器地址、模型资产路径、W&B 私有链接、日志样本或公司数据。
- 明确个人角色是复现、训练工程、调参与评测；没有证据时不认领上游代码。
- 若不确定某项结果能否披露，描述方法和经验，不给内部资产细节。
