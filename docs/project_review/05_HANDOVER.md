# TinyZero 离职交接与复现手册

## 1. 当前状态

- 本地仓库：`/Users/zihao.hu4/Documents/ChatGPT/tinyzero`
- 分支/提交：`main` / `95df88f`，与 `origin/main` 一致
- 工作树：生成本报告前干净；本次只新增 `docs/project_review/`
- 上游状态：README 已声明项目停止维护，推荐新实验迁移到最新 veRL
- 历史训练服务器：内部地址（公开版本已隐藏）
- 历史项目目录：`<PROJECT_ROOT>/TinyZero`
- 历史 conda 环境：`<CONDA_ENV>/tinyzero`
- 续训日志：`<PROJECT_ROOT>/logs/countdown-step300-continue-200.log`
- 本轮审计状态：服务器 SSH banner exchange 超时，远端路径需接手人重新确认

严禁在交接文档中保存 W&B API key、公司代理口令或其他凭据。此前会话曾暴露一枚 W&B key，应由账号所有者立即撤销并重新生成。

## 2. 环境与依赖

仓库 `requirements.txt` 声明：

```text
accelerate, codetiming, datasets, dill, flash-attn,
hydra-core, numpy, pandas, pybind11, ray,
tensordict<0.6, transformers<4.48, vllm<=0.6.3, wandb
```

README 推荐 Python 3.9、PyTorch 2.4.0/CUDA 12.1、vLLM 0.6.3。实际环境应以远端 `conda list --explicit` 或 `pip freeze` 为准；当前仓库没有 lock file，README 组合不等于已验证环境。

已知依赖冲突：`wandb==0.18.7` 安装后 protobuf 为 5.29.6，而 SwanLab 0.10.0 在 Python 3.9/Linux 要求 protobuf `>=6.31.1,<7` 且不等于 6.31.0。历史训练最终使用 W&B；不要在同一环境中无验证地强装 SwanLab。

## 3. 数据输入输出

### 3.1 原始数据

数据脚本读取 Hugging Face `Jiayi-Pan/Countdown-Tasks-3to4` 的 `train` split，期望字段：

```text
target: int
nums: list[int]
```

### 3.2 训练数据

默认输出：

```text
{DATA_DIR}/train.parquet  # 327680 rows
{DATA_DIR}/test.parquet   # 1024 rows，实际作为 validation
```

行字段：`data_source`, `prompt`, `ability`, `reward_model`, `extra_info`。详细格式见 `01_PROJECT_OVERVIEW.md`。

### 3.3 模型输出

期望：

```text
<think>自由文本推理</think>
<answer>(符合题意的四则表达式)</answer>
```

当前 scorer 只在输出最后一行搜索 `<answer>`，并要求所有给定整数恰好出现一次。

## 4. 关键入口

| 操作 | 入口 |
|---|---|
| 数据构建 | `examples/data_preprocess/countdown.py` |
| 默认训练 | `scripts/train_tiny_zero.sh` |
| PPO 主入口 | `verl/trainer/main_ppo.py` |
| 训练循环 | `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` |
| PPO/GAE/GRPO 数学 | `verl/trainer/ppo/core_algos.py` |
| Actor 更新 | `verl/workers/actor/dp_actor.py::DataParallelPPOActor.update_policy` |
| Countdown scorer | `verl/utils/reward_score/countdown.py::compute_score` |
| 数据加载 | `verl/utils/dataset/rl_dataset.py::RLHFDataset` |
| FSDP worker/checkpoint | `verl/workers/fsdp_workers.py` |
| 日志 | `verl/utils/tracking.py` |

## 5. 复现命令（仅文档，不在本次审计执行）

### 5.1 激活历史环境

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate <CONDA_ENV>/tinyzero
cd <PROJECT_ROOT>/TinyZero
```

登录时出现 `-bash: UID: readonly variable` 来自 shell 初始化文件错误地给 Bash 只读变量 `UID` 赋值。它通常不阻止后续命令，但应在个人 shell profile 中找到并删除/改名相关赋值；不要修改系统级文件，除非有授权。

### 5.2 数据构建

需要网络访问 HF 或预先下载的数据缓存：

```bash
python examples/data_preprocess/countdown.py \
  --local_dir <DATA_ROOT>/countdown \
  --train_size 327680 \
  --test_size 1024 \
  --template_type base
```

脚本没有主动创建 `local_dir`，运行前应确认目录存在。为避免覆盖已有资产，先 `ls`/`stat`，不要直接重跑。

### 5.3 官方默认训练

```bash
export N_GPUS=2
export BASE_MODEL=/absolute/path/to/Qwen2.5-3B
export DATA_DIR=<DATA_ROOT>/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
bash scripts/train_tiny_zero.sh
```

这是仓库默认命令，不等于历史实验精确命令。重跑前必须检查 GPU 数、模型路径、数据路径和 W&B 登录。

### 5.4 稳定性续训

历史覆盖参数为：

```text
actor_rollout_ref.actor.optim.lr=2e-7
algorithm.kl_ctrl.kl_coef=0.005
actor_rollout_ref.actor.entropy_coeff=0.001
trainer.test_freq=50
trainer.save_freq=50
trainer.total_training_steps≈201
```

恢复 checkpoint 的确切参数名/路径必须从历史启动命令或 resolved config 复制。当前代码的 `_save_checkpoint()` 只保存 actor/critic 到 `global_step_${global_steps}`，本版本是否原生自动恢复需核验；不要凭猜测拼接 resume 参数。

### 5.5 验证与严格评测

训练器会按 `test_freq` 自动调用 `_validate()`，结束时再做一次 final validation。严格 exact/format/3-number/4-number 评测所用脚本当前未在本地工作树中，必须从服务器归档后才能提供可信复现命令。

## 6. 必须归档的实验资产

离职前按优先级处理：

1. 历史初始训练与续训完整 stdout/stderr 日志；
2. step 300、续训 step 50/100/150/200 checkpoint manifest；
3. 最优 `global_step_150` actor 权重路径和文件 hash；
4. 每次实验的 resolved Hydra YAML；
5. W&B run URL、run ID、导出的 history CSV/JSON（不包含 API key）；
6. 严格评测脚本、命令、逐样本 predictions、metrics JSON；
7. base model 精确版本/commit/hash；
8. train/validation Parquet 的行数、schema 和 SHA256；
9. `pip freeze` 或 `conda env export --from-history`；
10. GPU 型号、数量、CUDA/driver、峰值显存与总训练时长。

建议目标目录结构：

```text
artifacts/
  manifests/
    environment.txt
    data.sha256
    checkpoints.sha256
  configs/
  logs/
  eval/
    predictions.jsonl
    metrics.json
  commands/
```

注意 checkpoint 很大，按公司资产策略保留在授权存储，不要擅自复制到个人设备。

## 7. 当前已知 Bug 与限制

1. `test.parquet` 实际用于 validation，存在选择偏差。
2. 数据按顺序切分，未做 canonical deduplication。
3. scorer 只解析最后一行的 `<answer>`，对尾部文本敏感。
4. scorer 用正则抽整数并用 `eval`；宜改 AST 白名单。
5. reward 0.1 混入 mean score，容易与 exact accuracy 混淆。
6. 没有基础模型 strict baseline。
7. 没有多 seed、置信区间和配对显著性检验。
8. 续训同时改变 lr/KL/entropy，无法做单变量归因。
9. 3 数/4 数难度差异很大，但旧 checkpoint 缺分组结果。
10. README 已标注仓库不维护，旧 vLLM/Transformers 依赖增加环境风险。
11. `countdown.py` 中 `gen_dataset()` 与多个 CLI 参数未被主流程使用，易误导。
12. 本地没有实验产物，证据链不完整。

## 8. 未完成实验

- 原始 base model 的同口径 strict baseline；
- 新建从未用于调参的 held-out test；
- 对旧/新 checkpoint 保存逐题结果并做 McNemar/paired bootstrap；
- PPO vs GRPO 同预算比较；
- lr、KL、entropy 单变量消融；
- 至少 3 个随机种子；
- 400 条正确推理数据的生成、去重、质量审计和冷启动 SFT；
- SFT 后再 RL 的增益与遗忘评估；
- 中间推理正确性/忠实性人工抽检；
- 推理吞吐、训练成本、GPU 利用率记录。

## 9. 推荐的下一步

### P0：证据闭环

先归档上述实验资产，并创建实验 manifest。没有这个步骤，任何后续提升都难以可信比较。

### P1：建立正确评测协议

从未被查看/训练的题池构造 held-out test；按 `(sorted(nums), target)` 分组去重；固定 greedy 与 sampling 两种评测；保存逐样本输出。对 base、old PPO、continued PPO 做一次同口径评测。

### P2：冷启动 SFT（若继续）

只能从 generation train pool 采样。建议 100–200 个 prompt，每题生成 4–8 个候选，经规则验证、去重、长度/模板/可读性过滤后选 400 条。再用 base model 做小规模 SFT，并在 held-out test 比较 base/SFT/PPO/SFT+RL。

### P3：算法对照

迁移到新 veRL，在相同生成 token 预算下比较 PPO 与 GRPO。对 GRPO 至少报告 group size、每题 reward variance、有效 group 比例与吞吐。

## 10. 安全与合规

- 不把公司代码、模型、数据、日志上传到公共网盘或个人 GitHub。
- 不在文档/脚本中硬编码 token、代理账号和密钥。
- 不使用无业务目的的矩阵计算长期占用 GPU；资源保活必须符合团队调度政策。
- 离职前确认模型和数据的许可证及公司内部成果边界。
- 对外表述只使用获准公开的信息；本报告默认是授权环境内的内部交接资料。

## 11. 接手人验收清单

- [ ] 能访问仓库与训练服务器
- [ ] 确认当前 Git SHA
- [ ] 找到 base model、数据和 checkpoint
- [ ] 校验所有 SHA256 与行数
- [ ] 激活环境并完成 import smoke test
- [ ] 从日志恢复完整启动命令/resolved config
- [ ] 在 16–32 条样本上只做离线推理与 scorer smoke test
- [ ] 复算 step150 metrics 与现有结果一致
- [ ] 建立新 held-out test
- [ ] 轮换已暴露的 W&B key
- [ ] 将远端脚本/日志纳入授权版本控制或制品库
