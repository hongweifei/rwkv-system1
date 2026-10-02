---
license: apache-2.0
language:
- zh
- en
- de
- fr
- es
- pt
- ru
- ar
- ja
- ko
- vi
tags:
- rwkv
- rwkv7
- classification
- calibration
- decision-model
base_model: RWKV/rwkv7-g1
library_name: rwkv
---

# RWKV-7 决策头（System 1 分类 / 打分模型）

中文 · [English](README_en.md)

一句话：**给它一段状态（一条工单、一段对话、一个告警），它在固定的候选项里选一个，并给出经过校准的概率分布**。它不生成文字。

底座是冻结的 [RWKV7-G1 "GooseOne" 0.1B](https://www.modelscope.cn/models/RWKV/rwkv7-g1)（Apache-2.0）。**它的权重原样打进了本包**，取下即用，不需要另外下载底座。权重文件发**两种容器**：`model.safetensors` 与 `model.pth`，装的是**同一批张量**——构建脚本把两份文件逐张量对一遍，任何一处不同就拒绝出包。默认读 safetensors（它连携带可执行对象的能力都没有），想要 `.pth` 就 `DecisionModel(".", fmt="pth")`。<!-- generated:counts -->（此段由构建脚本填充）<!-- end:generated -->

> 说清楚一件事：`model.pth` 里那 402 个张量是 RWKV 项目训出来的，不是这里训的。本包的作者只做它上面那些决策头（每个约 1 MB）。底座按 Apache-2.0 再分发，署名与出处见 [NOTICE.md](NOTICE.md)。
> 之所以敢这么打包：加载走的是和训练时**同一个**归一化函数（`rwkv7.normalise_native`），存进去的是**未加宽的原始张量**（402 个全是 bfloat16），bf16→fp32 与 ln0 融合发生在读取时、两条容器共用同一段代码。构建当场回读核对：每种容器 402 个张量逐位相同 + 前向零漂移，再互相比对两份文件；`config.json` 为每个容器和一个头都留了摘要，加载时先验再读。

## 30 秒上手

```bash
pip install -r requirements.txt
python examples/selfcheck.py        # 必须打印 OK: this model reproduces the numbers on its card.
```

```python
from modeling.service import DecisionModel

model = DecisionModel(".")                          # 默认读 model.safetensors
# 换容器：DecisionModel(".", fmt="pth")
# 也可以给仓库 id，数据自动下载进 SDK 缓存：
# model = DecisionModel("陆白tuber/rwkv7-system1")
print(model.tasks())                                # 它会回答哪些问题
r = model.score("laya_needs_human", '{"thread": [...对话记录...]}')
# {'answer': 'true', 'confidence': 0.170, 'probs': [0.262, 0.738], 'ranked': [...], ...}
```

`confidence` 用的是 Laya 的口径 `1 − H(p)/log k`（"我决定了多少"），不是最大概率；均匀分布时为 0，完全确定时为 1。

## 它能做什么：评测结果

**下表由构建脚本从产物里生成，不是手写的**（改代码重跑 `python scripts/build_modelscope_pkg.py` 就会刷新）。`acc` 与 `ECE` 都在 **T=1**（未做任何温度缩放）下测，测试集是各数据集自带的官方 test 切分。

<!-- generated:eval -->
（此段由 `scripts/build_modelscope_pkg.py` 填充；若你看到这行，说明包是从没有该占位符的 README 构建的。）
<!-- end:generated -->

## 它做不到什么（同样重要）

- **XNLI 句子推理八个语言 ≈ 瞎猜。** 三分类的基线是 33.3%，它在 34.0–42.0% 之间。数据加到 8 倍仍是平的，头部加宽也是平的：这是冻结底座把句子压成向量时丢了这个信息，不是训练没到位。
- **`boolq` 低于"永远答是"。** 59.17% 对多数类 61.7%，而且需要把温度放大到 T=2.0 才压得住它的过度自信。
- **它读不懂"评分标准文本"。** 头是从状态里挑位置，不是真的在按你写的规则判断：把训练好的选项顺序打乱，`ag_news` 的准确率从 87.5% 掉到 21.9%（换到别的头配置上可复现）。所以它适合**标签集合固定的分类任务**，不适合"任意 rubric 的判断器"。这一条是本仓库里所有准确率数字共同的前提。
- **MASSIVE 十种语言共用一个头**（`heads/rwkv7-g1d-0.1b_joint_massive_*.pt`）。所以表里那 10 行是同一个模型读十个测试集，不是十个独立模型；跨语言的提升来自共享，不是每种语言单独训了一份。

## 校准是怎么来的

两阶段训练（交叉熵预热 → 严格 proper 评分 RLCD），再在 dev 上给每个 `(问题类型, 选项数)` 桶拟合一个温度标量；拟合受"一个标准误"规则保护——只有 dev 分辨得出来的缩放才会被发布。表里 `Tfit=1.00` 的任务就是拒绝被缩放的那些。dev 一律取数据集自带的 validation 切分，绝不用测试集调任何东西。

## 训练设置（可复现）

| | |
|---|---|
| 底座 | `rwkv7-g1d-0.1b-20260129-ctx8192.pth`，冻结，一次前向抽特征 |
| 头 | 逐选项打分 MLP，宽度 256，无 set attention、无槽位嵌入，约 26 万参数 |
| 配方 | CE 预热 60 epoch → RLCD 30 epoch → 温度拟合；lr 3e-3，batch 64，AdamW wd 1e-2 |
| 数据 | 34 个任务，每任务 225–16000 行训练样本（详见表里 `n_test`） |
| 硬件 | 纯 CPU（PyTorch，无 CUDA）；一个头约 45 秒 |
| 种子 | <!-- generated:seeds -->（此段由构建脚本填充）<!-- end:generated --> |

复现整条链路：

```bash
python -m rwkv_system1.pipeline extract --tasks <task>   # 抽特征（CPU，约 0.7 s/条）
python -m rwkv_system1.pipeline train --tasks <task>
python -m rwkv_system1.pipeline eval --tasks <task>
```

## 许可与来源

- 本包的代码与决策头：Apache-2.0（见 [NOTICE.md](NOTICE.md)）。
- 底座权重与分词表来自 **RWKV 项目**（`RWKV/rwkv7-g1`，Apache-2.0）。底座权重**原样打进两个权重文件一起再分发**，按该许可证署名；它不是本包训练的，见 [NOTICE.md](NOTICE.md)。
- 评测数据来自公开数据集（Laya typed-decisions、MASSIVE、XNLI、Banking77、go_emotions、amazon_reviews_multi、ag_news、sst5、boolq 等），各自保留原许可；引用它们的原始结果时请注明数据集出处。

## 这个模型是怎么被验证的

发布前，构建脚本会做下面这些事，任何一件不过就**拒绝出包**：

0. **容器核对**：每种容器写完后读回来，要求那 402 个底座张量与直接读上游文件**逐位相同**、同一次前向 `max|Δhidden| = 0`；再把 `.pth` 与 `.safetensors` **互相比对**，张量层面一处不同就不出包。`config.json` 记下两个文件各自的 sha256 和**每个头的摘要**，加载时先验再读。
1. 用包里的头 + 缓存特征重跑该任务官方 test 集，要求 **accuracy / ECE / NLL 与上表逐位相同**（不是"接近"）；集成版本这里是拿**三个头的平均分布**去量的；
2. 检查该任务缓存记录里的 `marker_rule` 与注册表解析出的取特征规则一致（`last` 与 `pooled` 的向量数值差约 1e2，装错会静默算错）；
3. 取一条真实样本走完整"实时"路径（分词 → prefill → 选项块 → 取标记 → 头），要求与缓存路径选出同一个答案。

`examples/selfcheck.py` 在任何机器上重放第 3 步，并把概率偏差报出来（实测最大 1e-4，容差 0.05，余量来自批形状不同导致的浮点归约顺序差异）；它还会把两种容器互相比一遍（34 个任务上差值 0.0e+00）。这些闸门不是摆设：只往其中一种容器里改一个头的一个权重，加载就会拒绝，并点名是哪个头、摘要差在哪。
