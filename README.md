# rwkv-system1

中文 · [English](README.en.md)

冻结的 RWKV-7 底座 + 逐选项打分头：**给它一段状态（一条工单、一段对话、一个告警），它在固定的候选项里选一个，并给出校准过的概率分布**。它不生成文字。

模型本身怎么看、怎么用、哪里不行：[模型卡 `packaging/modelscope/README.md`](packaging/modelscope/README.md)（出包后就是魔搭页面上那张卡）。**本文件讲这个仓库**：代码怎么组织、一条实验怎么跑、产物为什么不进 git、出货前要过哪几道闸门。

## 它是什么

| | |
|---|---|
| 底座 | `rwkv7-g1d-0.1b`（12 层、d_model 768、vocab 65536），**全程冻结**，一条样本一次前向抽特征 |
| 头 | 逐选项打分 MLP，宽 256，无 set attention、无槽位嵌入，约 26 万参数 |
| 配方 | CE 预热 60 epoch → RLCD 30 epoch → 温度拟合；lr 3e-3，batch 64，AdamW wd 1e-2 |
| 集成 | 3 个种子，取 **softmax 概率的均值**（不是 logit 均值） |
| 共享头 | MASSIVE 十个语言共用一个头（`joint_massive`） |
| 硬件 | 纯 CPU（PyTorch，无 CUDA 依赖） |

## 现在的成绩（34 个任务，各数据集自带的官方 test 切分）

| | |
|---|---|
| 套件均值 accuracy | **52.04%** |
| 最好的几个 | `prompt_injections` 97.4% · `ag_news` 91.6% · `laya_category` 81.0% · `banking77` 76.8% |
| MASSIVE 十语均值 | **53.3%**（该任务在 60 个意图里选一个，均匀瞎猜 ≈ 1.7%） |
| 几乎没有信号 | XNLI 八语 34.0–39.7%（三分类基线 33.3%） |
| 低于"永远答多数类" | `boolq` 59.83% 对 61.67% |

数字读自 `experiments/results/eval_shipped_ens3.json`（出货那张表）。逐任务的 acc / ECE / NLL / 拟合温度在模型卡的生成表里，那一栏由构建脚本从产物算出，不手写。

## 把仓库跑起来

```bash
git clone <本仓库地址> rwkv-system1 && cd rwkv-system1
uv sync                                          # Python 3.14+，torch 2.14 CPU 版

uv run python tests/invariants.py                # 12 项自检，秒级：不碰网络、也不碰权重
```

自检在**全新 clone** 里也能全绿：其中两项会算"特征身份"，而身份里含底座文件的摘要——那两项自带占位文件，所以不会因为少了几百 MB 权重就变成假绿（详见 `tests/invariants.py` 的 `stand_in_weights`）。

要真跑训练/评测，还缺底座：

```bash
uv run modelscope download --model RWKV/rwkv7-g1 \
  --include "rwkv7-g1d-0.1b-20260129-ctx8192.pth" --local-dir weights
```

（这条命令就是 `cache_layout.checkpoint()` 在找不到文件时打印的那一条。）

## 一条完整链路

```bash
python -m rwkv_system1.pipeline data     --tasks banking77   # 先看渲染出来的提示词长什么样
python -m rwkv_system1.pipeline extract --tasks banking77   # 抽特征 → cache/，约 0.68 s/条
python -m rwkv_system1.pipeline train   --tasks banking77 --ensemble-seeds 3
python -m rwkv_system1.pipeline eval    --tasks banking77 --ensemble-seeds 3
python -m rwkv_system1.pipeline predict --tasks boolq --n 3  # 实时路径：分词→prefill→选项块→取标记→头
python -m rwkv_system1.pipeline ablate  --tasks banking77     # 一次前向、多个头，比配方
```

一个头约 45 秒；一次全量抽特征按任务大小从几分钟到几小时。`--feature-rule {last,pooled,two_layer}`、`--n-train/-dev/-test`、`--targets {gold,crossfit,teacher,hard}` 这些开关**会换产物名字**，不是换行为而已。

## 目录

| 目录 | 里面是什么 |
|---|---|
| `rwkv_system1/` | 包本体。`backbone.py` 模板与 prefill；`features.py` marker→向量与 gather 规则；`cache_layout.py` **只负责名字**；`train.py` / `rlcd.py` / `decision.py` 训练与打分；`rwkv7.py` 自己写的 RWKV-7 前向（`scripts/verify_rwkv7.py` 拿它和官方 `rwkv` 包在 CPU 上对过）；`datasets/` 每个数据集一个 loader；`pipeline/` CLI 与五个命令 |
| `experiments/` | `harnesses/` 一段实验一个文件，`chains/` 把实验串起来并**逐步打印 `*_RC=`**，`results/` 是留在库里的证据（JSON 结果表 + TXT 小结） |
| `packaging/modelscope/` | 出包时**由人手写**的部分：模型卡（中 / 英）、`NOTICE.md`、`modeling/service.py`（发布后用户 import 的那层薄服务）、`examples/selfcheck.py` |
| `scripts/` | `build_modelscope_pkg.py` 组装 + 闸门；`upload_modelscope.py` 发布；`smoke_train.py` 不抽特征先验头；`verify_rwkv7.py`；`prune_caches.py` |
| `tests/invariants.py` | 12 项不变量，每一项对应一次真实事故 |

## 哪些东西不在 git 里，为什么

判据只有一条：**能用「库里的东西 + 一条命令」重算出来的，不进历史。**

| 不在库里 | 怎么再来 | 代价 |
|---|---|---|
| `weights/` | 上面那条 `modelscope download` | 网络 |
| `cache/` | `pipeline extract` | 约 0.68 s/条，最贵的一项 |
| `runs/` | `pipeline train` | 一个头约 45 s |
| `dist/` | `python scripts/build_modelscope_pkg.py` | 十几分钟（含闸门） |
| `.venv/`、`refs/`、`.env` | `uv sync`、包索引、你自己的密钥 | — |

`packaging/modelscope/` 里 authored 的部分**是**跟踪的，`dist/` 只是它的组装副本。发布包（`model.pth` + `model.safetensors`）里**打了底座的 402 个张量**并按 Apache-2.0 再分发；**仓库自己不携带底座权重**，两者别混。

## 名字的规矩（这仓库最容易出事的地方）

一个缓存的文件名编进了它的全部输入：模板、token 预算、gather 规则、分词表与底座文件的摘要、`data_version`、三个切分的大小。头的名字再叠上配方旋钮（宽度、层数、槽位、是否重排选项、targets、温度拟合、seed……）。于是：

- 改其中任何一项 = 换名字 = 旧产物**自动失效**，而不是被静默复用；
- **名字必须描述产物，不能描述请求**——曾经有名字写着 `n11000`、文件里只有 10930 行；
- 存在产物里的那份 identity 和名字里那个哈希，必须永远出自同一个函数（这条是六次同一类 bug 的共同修法）。

一个会咬人的细节：缓存**名字**里的哈希**不含** `data_version`（版本号单独拼成 `_d2`），而 sidecar 记录**含**；头的 `feature_inputs` 又是按记录算的，共享头则是整组记录的哈希。所以"判断某个产物还可不可达"必须**两种基准都对得上**才算，只用一种会把 393/407 个健康文件判成死文件——这事真发生过。

## 出货前的闸门

`python scripts/build_modelscope_pkg.py` 对每个任务做三件事，任何一件不过就**拒绝出包**：

1. 用包里的头 + 缓存特征重跑该任务官方 test，要求 **accuracy / ECE / NLL 与卡片逐位相同**（不是"接近"）；
2. 检查缓存记录里的 `marker_rule` 与注册表解析出的取特征规则一致（`last` 与 `pooled` 的向量数值差约 1e2，装错会静默算错）；
3. 取一条真实样本走完整"实时"路径，要求与缓存路径选出同一个答案。

再加一道容器间的：`model.pth` 与 `model.safetensors` 装的是同一批张量，构建时逐张量比对，不同就拒绝。

在**出好的包里**（`cd dist/modelscope`）跑 `python examples/selfcheck.py`，它会在任何机器上重放第 3 步并报出概率偏差。`scripts/upload_modelscope.py` 在自检不通过时**拒绝联网**；token 只从环境变量或 `.env` 读（`.env` 已在忽略列表，绝不进命令行）；推完回读远端文件清单，缺一个文件就报失败——半截上传不能长得像成功。

## 许可与来源

- 本仓库的代码与决策头：Apache-2.0。署名与第三方来源清单见 [`packaging/modelscope/NOTICE.md`](packaging/modelscope/NOTICE.md)。仓库里**目前没有 LICENSE 原文文件**，需要的话补一份 Apache-2.0 正文即可，代码与卡的声明都以它为准。
- 底座权重与分词表来自 **RWKV 项目**（`RWKV/rwkv7-g1`，Apache-2.0）：仓库不携带它，发布包按 Apache-2.0 再分发并注明出处。
- 评测数据来自公开数据集（Laya typed-decisions、MASSIVE、XNLI、Banking77、go_emotions、amazon_reviews_multi、ag_news、sst5、boolq 等），各自保留原许可；引用它们的原始结果时请注明数据集出处。
