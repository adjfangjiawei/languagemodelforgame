# 中国象棋博弈图（Variation-DAG）思维链对话模型 — GTX 1660 Ti (6GB) ~4小时训练完整工程包

本工程针对 **NVIDIA GeForce GTX 1660 Ti（6 GB 显存，Turing `sm_75` 架构）** 量身定制，内置 **Pikafish（皮卡鱼 2026-09 NNUE 大师/特级大师级引擎，含 Windows `.exe` 与 Linux 双平台二进制及 `pikafish.nnue` 权重）**、**10,240 词表通用中英+象棋无损 Byte-Level BPE 分词器**、**五重可读中文博弈图（Variation-DAG）CoT 合并器** 与 **FP16 + 分块 FP32 交叉熵显存优化训练器**。

解压后无需手工下载任何引擎或词表，**双击/一行命令即可在约 4 小时内完成多核大规模数据构建 + GPU 收敛训练 + 自动评测**，并支持后续无缝扩展训练通用中文自然语言对话与文本。

---

## 一、快速上手（约 4 小时一键全流程）

### 1. Windows（GTX 1660 Ti 本地推荐）
1. 解压 `xiangqi_1660ti_gpu.zip` 到任意英文/中文路径。
2. 确保已安装 Python 3.10+ 及 CUDA 版 PyTorch（若未安装，`run_4h_windows.bat` 会自动检测并安装 CUDA 12.1 版 PyTorch）：
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. 直接双击运行 **`run_4h_windows.bat`**，或在终端执行：
   ```powershell
   python run_4h_pipeline.py --total-hours 4.0 --data-minutes 35 --preset fast-4m
   ```

### 2. Linux / WSL2
```bash
bash run_4h_linux.sh
```

### 3. 4 小时时间预算自动分配机制
- **阶段一（约 35 分钟上限，多核 CPU 并行）**：调用 `scripts/build_dataset_parallel.py` 跨全部 CPU 核心并行对弈采样、严格镜像去重、执行 4~5 步分数阈值与防震荡搜索，并通过 **Variation-DAG 同形合流** 生成 `1024` 长度预算内的高密度可读中文思维链数据（在 6~8 核桌面/笔记本 CPU 上约 20~35 分钟即可生成 **30,000 ~ 45,000 条** 训练样本，支持断点续跑 `--resume`）。
- **阶段二（约 3.4 小时上限，GTX 1660 Ti FP16 高吞吐训练）**：调用 `train_gpu.py` 训练默认的 **`fast-4m`（413 万参数，比 `base-45m` 小 10.7 倍！）** 高速模型。结合**低秩分解词嵌入（`d_embed=128`）+ 2:1 GQA + 长度分桶采样（消除 ~40% `<pad>` 冗余计算）+ 单次融合 GEMM 交叉熵**，在 GTX 1660 Ti 上仅占用 **~0.85 GB 显存**，4 小时内可轻松跑完 **30,000 ~ 40,000+ Steps**（甚至提前收敛完成）。
- **阶段三（约 2 分钟）**：自动在独立验证集上评测 `<|think|>...<|/think|>` 格式率、同形合并率、单次贪婪合法率与 Best-of-4 引擎可行着法命中率，输出 `outputs/eval_metrics.json` 与损失曲线 `outputs/training_curves.png`。

---

## 二、硬件加速与十倍模型瘦身架构（专为 GTX 1660 Ti 6GB `sm_75` 设计）

| 优化项 | 技术细节 | 实际收益 |
| :--- | :--- | :--- |
| **低秩分解词嵌入 (`d_embed < d_model`)** | 将 `10,240` 大词表投影解耦为 `10240 -> 128 -> 256`（首尾权重绑定） | 词表参数与输出层 `lm_head` 计算量直降 **4 倍**，同时保留 `d_model=256, head_dim=64` |
| **分组查询注意力 (2:1 GQA)** | `n_heads=4 (head_dim=64), n_kv_heads=2`，完美对齐 CUDA SDPA 快路径 | 降低 KV 投影参数、显存带宽与推理 KV-Cache 开销 |
| **长度分桶采样 (`LengthBucketBatchSampler`)** | 按序列长度分桶组装 Micro-Batch 并打乱批次顺序 | 消除约 **40%** 的 `<|pad|>` 无效填充计算，吞吐直接提升 **~1.65x** |
| **Turing `FP16 + GradScaler` + 原生半精度 RoPE** | `FP32SafeRMSNorm` 防溢出，RoPE 保持原生 FP16 零类型往返拷贝 | 杜绝 NaN 的同时跑满 GTX 1660 Ti FP16x2 吞吐 |

### 可选模型档位（均满足 `>= 100 万参数` 且支持命令行任意微调维度）
- `--preset nano-2m`：**2.59M 参数**（`d_embed=96, d_model=192, L=4, H=3/1`，比 `base-45m` 小 **17.1 倍**，极致飞快）
- `--preset fast-4m`（**默认推荐**）：**4.13M 参数**（`d_embed=128, d_model=256, L=4, H=4/2, d_ff=640`，比 `base-45m` 小 **10.7 倍**，速度与推理能力最佳甜点）
- `--preset light-6m`：**5.80M 参数**（`d_embed=128, d_model=256, L=6, H=4/2, d_ff=704`，比 `base-45m` 小 **7.6 倍**）
- `--preset small-9m`：**9.36M 参数**（`d_embed=160, d_model=320, L=6, H=5/5, d_ff=896`，比 `base-45m` 小 **4.7 倍**）
- `--preset small-16m`：**18.1M 参数**（`d_model=384, L=8, H=6`）
- `--preset base-45m`：**44.3M 参数**（`d_model=512, L=12, H=8`，适合多卡或更长多日训练）
- `--preset large-88m`：**88.4M 参数**（`d_model=640, L=16, H=10`）
- 亦可直接用 `--d-model`、`--d-embed`、`--n-layers`、`--n-heads`、`--n-kv-heads`、`--d-ff` 自定义任意尺寸！

---

## 三、五重可读中文 CoT 合并机制（完整保留中文连接词）

详见 `xiangqi/dag_cot.py` 与基准测试脚本 `scripts/demo_dag_compression.py`：
1. **【S1 前缀字典树合并】**：共享前序着法不重复叙述，叶节点合并为 `此后红可走E或F`。
2. **【S2 跨分支/子分支同局面合流（Exact Transposition DAG）】**：每一步精确记录落子后盘面键 `"{grid_fen} {turn}"`。当不同走子次序（如 `A B C` 与 `H I J`）到达同一盘面且满足前缀拆分净收益判据时，自动合并前置条件：
   > `【同形合流1】如果黑走炮8平7、红应马八进九、黑续象3进5，或者黑走象3进5、红应马八进九、黑续炮8平7，则双方归于同一盘面，若红走车一平二则黑可走车9进1、士4进5或炮2进2……`
3. **【S3 兄弟子树同构后缀归并（Merkle Isomorphism）】**：即使盘面 FEN 不同（如 `士4进5` 与 `士6进5`），若后续推演子树文本完全一致，自动合并为 `黑若应士4进5或士6进5，均……`。
4. **【S4 同子多落点自然省主语归并】**：遵循传统中文棋谱习惯，将 `炮8平6、炮8平7、炮8平9、车9进1` 自然表述为 `炮8平6、平7、平9或车9进1`。
5. **【S5 唯一应着线性顺承 + 最优前缀拆分代价门控】**：单分支强制应手直接顺承；仅在同形后缀节省 Token 大于打断前缀树开销时抽提合流，确保在 6 类典型实战局面上相比平铺枚举实现 **5.58x ~ 9.66x 的 Token 压缩比**，`1024` 上下文单条容纳可行路线数从 **24~28 条提升至 115~172 条**。

运行压缩率实测对比：
```bash
python scripts/demo_dag_compression.py
```

---

## 四、10,240 通用中英+象棋词表与第二阶段通用自然语言训练

`tokenizer_assets/tokenizer_10k.json`（精确词表大小 **`10,240`**）采用 **GPT-4 / Qwen2 同源 Byte-Level BPE** 架构：
- **零 `<unk>` 无损编解码**：256 个基础字节全覆盖，任意生僻汉字、古文、英文、Python/C++ 代码、数学符号与 Emoji 均可 100% 无损还原。
- **全面覆盖通用中文**：内置全部 **GB2312 一级+二级常用汉字（6,763 字）** + 高频现代汉语双字/多字词与成对连接词（`如果`、`或者`、`因为`、`所以`、`虽然`、`但是`）+ 常用英文词根 + 象棋着法原子词，实测通用中文约 **`0.88 token/字`**，英文/代码 **`0.42 token/字符`**，象棋 DAG-CoT **`0.46 token/字`**。

### 如何继续训练通用自然语言（两种方式均已原生支持）

1. **方式 A：混合联合训练（防止遗忘，一步到位）**
   准备通用中文对话/文本 JSONL（支持 `{"messages": [...]}`、`{"prompt": "...", "response": "..."}` 或纯文本 `{"text": "..."}` 三种主流格式），直接传入 `--general-jsonl`：
   ```bash
   python train_gpu.py --preset fast-4m --general-jsonl path/to/general_chinese_sft.jsonl --max-hours 4.0
   ```
2. **方式 B：两阶段课程/持续训练（先注入象棋推理 CoT，再训通用语料或反向注入）**
   ```bash
   # 在已完成象棋 DAG-CoT 训练的权重基础上，加载通用语料继续交叉熵训练：
   python train_gpu.py --preset fast-4m --init-from checkpoints/best_model.pt --general-jsonl path/to/general_chinese.jsonl --lr 3e-4
   ```

---

## 五、可视化网页对弈与交互命令（支持边训练边热重载看 CoT！）

### 1. 启动直观的可视化对弈网页（零额外依赖）
- **Windows**：直接双击 **`start_web_windows.bat`**（自动弹出浏览器打开 `http://127.0.0.1:7860`）
- **Linux / 命令行**：
  ```bash
  python web_app.py --port 7860 --open-browser
  ```
- **网页功能亮点**：
  - 点击己方棋子自动高亮全部合法落点与吃子框，点击落子后 AI 自动在右侧**实时流式输出 `<|think|> ... <|/think|>` 思维链**并走子回应；也可直接点 **「🤖 让 AI 思考并走当前步」** 让 AI 执红先走或连续互弈。
  - 支持在 **`train_gpu.py` 后台训练的同时** 打开网页，随时点击右上角 **「🔄 热重载最新权重」** 立刻体验最新 `best_model.pt` 的棋力与思维链进化！
  - 内置三种推演模式切换：`自研神经网络 CoT（智能合法守护）`、`纯自研神经网络 CoT`、`Pikafish 大师引擎 + 五重【同形合流】DAG-CoT`。

### 2. 终端命令行推理与验证集评测
```bash
# 1. 启动交互式命令行（支持直接输入如 `炮二平五` 对弈、`fen <FEN>` 分析任意残局、以及通用中文问答）
python inference_chat.py --interactive

# 2. 对指定 FEN 盘面输出完整 <think>...</think> 推演与最佳走法
python inference_chat.py --fen "r1bakab1r/9/1cn4c1/p1p1p1p1p/9/2P6/P3P1P1P/1C2C1N2/9/RNBAKAB1R w - - 0 5"

# 3. 批量验证集定量评测
python inference_chat.py --eval-val --num-eval 100
```

---

## 六、增量继续生成 CoT 数据 + 强制指定 Step 数续训（不走时间额度，支持千万级规模）

当你希望不断追加生成新的象棋盘面 CoT 数据、并按**精确指定的 `step` 数**（完全不走时间额度）持续续训时：

### 1. 一键「追加新 CoT 数据 + 强制追加 Step 续训」
```bash
# 追加生成 10,000 个全新不重复盘面（约 20,000 条左右镜像 DAG-CoT），并在当前 checkpoint 基础上强制再训 +20,000 steps（无时间限制！）
python continue_sft.py --add-positions 10000 --add-steps 20000

# 若使用千万级规模分片存储（例如每 200,000 条样本自动切分一个 train_shard_XXXX.jsonl）：
python continue_sft.py --add-positions 100000 --shard-size 200000 --add-steps 150000
```
- **千万级去重索引 (`data/seen_canonical_u64.bin`)**：每个已生成的对称等价盘面仅占 8 字节 64-bit BLAKE2b 指纹，即使累积到 **1,000 万个盘面仅占 80 MB**，断点续生加载耗时 `< 0.2s`，保证新增盘面与历史数据零重复。
- **增量磁盘 `np.memmap` 分词缓存 (`*.tokcache.{bin,idx.npy,meta.json}`)**：续训时只对 JSONL **新追加的尾部行**做增量分词并追加写入二进制磁盘映射文件，千万条 CoT 数据训练时内存占用依然极低。
- **分段重预热余弦退火 (`Continual Cosine Schedule`)**：使用 `--add-steps K` 时，优化器自动在 `[start_step, start_step + K]` 区间进行短预热与平滑余弦衰减，避免旧计划衰减到 `min_lr` 后新数据学不动。

### 2. 单独调用数据扩充或强制 Step 训练
```bash
# 仅追加生成 +5,000 个全新盘面 CoT（无时间上限）
python scripts/build_dataset_parallel.py --add-positions 5000 --workers 6

# 从头或断点强制训练到精确总步数 50,000 step（完全不走时间额度）
python train_gpu.py --preset fast-4m --steps 50000 --resume

# 在已有 latest_checkpoint.pt 基础上强制额外再训 +15,000 step（完全不走时间额度）
python train_gpu.py --preset fast-4m --resume --add-steps 15000
```

---

## 七、后训练强化学习探索 CoT（双模式：AlphaGo 完整对局自我对弈 + DeepSeek-R1/DAPO 神谕探索）

当模型经过大规模（如数百万至上千万条局面）交叉熵 SFT 掌握了可读中文思维链与基础棋感后，工程同时提供**两套互补的后训练强化学习引擎**（均可单独运行或串联运行，强制精确 `step` 控制、不走时间额度）：

### 方案 A（核心象棋自博弈）：AlphaGo / AlphaZero 风格完整对局【自我对弈】CoT 强化学习（`train_rl_selfplay.py`）
把模型直接放入真实红黑象棋对局中**自己与自己（及历史冠军池）完整下棋**，全程 **0 外部引擎依赖**，完全依靠真实对局胜负演进 `<|think|>` 思维链与落子能力：

```bash
# Windows 双击 run_selfplay_rl_windows.bat，或命令行运行：
python train_rl_selfplay.py \
  --init-from checkpoints/best_model.pt \
  --steps 2000 \
  --games-per-step 4 \
  --cot-rollouts 2 \
  --opponent-mode league \
  --sync-best-to-main

# 断点继续强制追加 +1,500 Steps 自我对弈强化学习（无时间上限）：
python train_rl_selfplay.py --resume --add-steps 1500 --sync-best-to-main
```

- **多盘并行自博弈 (`generate_batch`)**：每步在 GPU 上并行推进多盘完整红黑对局（左填充 + 精确逐行 RoPE 位置编码），每一步棋都真实先生成 `<|think|> ... <|/think|>\n最佳走法：a_t` 再落子推进棋局。
- **AlphaZero 双头架构 (`Policy + AlphaGoValueHead`) + 步内 `K-CoT` 前瞻选优**：
  - 在共享 Transformer 末尾挂载轻量级局面价值头 $V_\phi(s_t) \in [-1, +1]$，训练时与策略头共享**同一次前向传播**。
  - 每步可采样 `K` 条候选思维链，模拟落子后用自身价值头评估后继盘面 $-V_\phi(s'_k)$（开局阶段叠加 AlphaGo 狄利克雷探索噪声 $\text{Dir}(\alpha)$），选出更强着法落子。
- **完整中国象棋纯规则裁判（0 外部引擎）**：精确判定绝杀、**困毙判负**（中国象棋专属规则）、**长将判负**、三次重复不变作和、双方无进攻子力作和与 60 回合自然限着，产出真实终局胜负 $z \in [-1, +1]$。
- **零和交替回合 Negamax GAE + 语言可读性守护**：利用双人零和博弈符号翻转 $\delta_t = \gamma (-V_\phi(s_{t+1})) - V_\phi(s_t)$ 结合真实终局胜负 $z_t$ 反向强化整局的 `<|think|>` 推理链与着法；同时通过纯规则 CoT 自洽检测 + 冻结 FP16 参考策略 $\pi_{\text{ref}}$ 的无偏 KL 惩罚，保证长周期自博弈中**绝不丢失自然中文连接词与可读性**。
- **AlphaGo 冠军擂台赛晋升机制 (`Champion Arena` + `League Pool`)**：每隔 `--arena-interval` 步自动举行先后手交换的对称擂台赛（`Learner` vs `Champion`），计算精确胜率与 $\Delta\text{Elo}$，胜率达标即加冕新冠军 `checkpoints/best_selfplay_model.pt` 并加入历史联盟池防止策略循环遗忘。

---

### 方案 B（保留单步多分支探索）：DeepSeek-R1 / DAPO 风格 Pikafish 可验证神谕 GRPO（`train_rl_grpo.py`）

若希望利用 Pikafish NNUE 对单盘面采样 `G` 条不同 CoT 做细粒度子力悔值 $\Delta_{\text{cp}}$ 校准，可运行保留的 `train_rl_grpo.py`（`run_rl_windows.bat`）：

```bash
# 基于 SFT（或自对弈后）权重运行 3,000 Steps 的 GRPO 强化学习思维链校准
python train_rl_grpo.py \
  --init-from checkpoints/best_model.pt \
  --steps 3000 \
  --group-size 6 \
  --prompts-per-step 2 \
  --temperature 0.85 \
  --kl-beta 0.02 \
  --sync-best-to-main

# 断点继续追加 +2,000 Steps 强化学习探索：
python train_rl_grpo.py --resume --add-steps 2000 --sync-best-to-main
```

- **四维可验证奖励 (`VerifiableXiangqiRewardEngine`)**：综合 `R_format`（格式与中文连接词可读性）、`R_legal`（规则合法性）、`R_strength`（Pikafish 分差悔值 $\Delta_{\text{cp}}$）与 `R_cot_process`（思维链自洽与同形合流归纳）。
- **DAPO 非对称裁剪 (`eps_low=0.20, eps_high=0.28`) + 零方差动态重采样 + 冻结 FP16 参考模型无偏 KL 约束**，支持千万行分片 JSONL 流式读取。

---

## 八、可视化网页人机对弈与实时 CoT 推演台

```bash
# Windows 双击 start_web_windows.bat，或命令行运行：
python web_app.py --host 127.0.0.1 --port 7860
```
Web 端会自动按优先级识别并加载 `best_selfplay_model.pt`（显示 `[AlphaGo自对弈]` 及对局数/`ΔElo`）、`best_rl_model.pt`（`[RL-GRPO]`）或 `best_model.pt`（`[SFT]`），训练期间随时点击 **「热重载最新权重」** 即可实战对弈。

---

## 九、目录结构说明

```text
xiangqi_1660ti_gpu/
├── engine/                        # Pikafish Windows (.exe) + Linux 二进制 + pikafish.nnue 权重 + 开源协议
├── tokenizer_assets/
│   └── tokenizer_10k.json         # 预构建 10,240 词表通用中英+象棋无损 Byte-Level BPE 分词器
├── xiangqi/
│   ├── board.py                   # 完整中国象棋规则、合法着法生成、中文记谱双向转换、盘面等价键 state_key
│   ├── engine_wrapper.py          # Windows/Linux 自适应 Pikafish NNUE UCI 封装
│   ├── tree_search.py             # 带置换表缓存的 4~5 步分数阈值与防震荡可行博弈树搜索
│   └── dag_cot.py                 # 五重可读中文博弈图（同局面合流 + 前缀树 + 同构子树 + 同子落点）生成器
├── model/
│   ├── tokenizer.py               # 10,240 通用+象棋分词器封装
│   ├── transformer.py             # Factorized Embedding + GQA + SwiGLU + 多局并行自博弈 generate_batch
│   └── dataset.py                 # 千万级分片流式加载 + 增量磁盘 np.memmap 分词缓存数据集
├── scripts/
│   ├── build_dataset_parallel.py  # 增量追加 (--add-positions) + 64-bit 指纹去重 + 分片 (--shard-size) 构建器
│   ├── demo_dag_compression.py    # 同形合流与五重合并策略实测 Token 压缩率基准测试
│   └── train_tokenizer.py         # 10,240 词表自定义重训脚本（支持挂载额外通用语料 --extra-corpus）
├── web_ui/
│   └── index.html                 # 交互式 SVG 棋盘 + 流式 CoT 可视化前端
├── train_gpu.py                   # 混合精度 SFT 主程序（支持 --steps / --add-steps 强制精确步数，无时间限制）
├── continue_sft.py                # 一键增量生成新 CoT 数据 + 强制指定步数续训编排器
├── train_rl_selfplay.py           # 【核心】AlphaGo 风格完整红黑自我对弈 CoT 强化学习主程序（0 引擎依赖 + 价值头 + 冠军擂台）
├── train_rl_grpo.py               # 【保留】DeepSeek-R1 / DAPO 风格单步多采样可验证奖励 GRPO 强化学习主程序
├── web_app.py                     # 零依赖 Web 对弈与流式思维链服务（自动识别 AlphaGo自对弈 / GRPO / SFT 权重）
├── inference_chat.py              # 交互对弈、单盘面推演与验证集评测 CLI
├── run_4h_pipeline.py             # 一键全自动编排脚本
├── continue_sft_windows.bat       # Windows 一键增量生数 + 精确步数续训脚本
├── run_selfplay_rl_windows.bat    # Windows 一键 AlphaGo 完整对局自我对弈 RL 脚本
├── run_rl_windows.bat             # Windows 一键 GRPO 强化学习后训练脚本
├── start_web_windows.bat          # Windows 一键启动网页对弈界面脚本
└── requirements.txt               # Python 依赖清单
```
