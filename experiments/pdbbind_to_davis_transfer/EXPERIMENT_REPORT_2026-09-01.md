# PDBbind → Davis 细粒度迁移：fold1 冻结局部特征试验

日期：2026-09-01  
服务器目录：`/data1/ztx/MyModel-MDTA/experiments/pdbbind_to_davis_transfer/`

## 结论

本试验的 go/no-go 结果为 **NO-GO**。PDBbind 预训练表示确实学到了非平凡、原子敏感的 atom–residue 结构，并且新的无泄漏源模型在 PDBbind 上优于旧模型；但将该表示冻结后作为 Davis fold1 全局模型的残差输入，验证 MSE 不降反升，且明显不如随机同构编码器。因此不应直接扩展到五折，也不应在当前接口上解冻 PDBbind 编码器。

关键 Davis fold1 验证结果（3 seeds）：

| 条件 | MSE mean ± std | 改善种子数 |
|---|---:|---:|
| G0 冻结全局基线 | 0.489465 | — |
| G1 全局 affine 校准 | 0.490584 ± 0.000628 | 0/3 |
| R0 随机同构局部编码器 | 0.487733 ± 0.002081 | 3/3 |
| P0 PDBbind 预训练局部 | 0.492517 ± 0.001125 | 0/3 |
| P-shuffle 原子特征打乱 | 0.497070 ± 0.001649 | 0/3 |
| P-mismatch 错配配体 | 0.491940 ± 0.002230 | 1/3 |

P0 相对全局基线的 MSE “改善”为 -0.6234%，未达到预设的 +2% 门槛；P0 也没有优于 affine 或 random。Davis test 指标未计算，训练/选择仅访问 fold1 train 20,774 行和 validation 3,094 行。

## 防泄漏审计

原始 PDBbind dual-cold 训练集与 Davis 存在跨数据集化学重合：fold1 validation 药物 `447077` 有精确结构重合，fold1 test 药物 `44259` 有骨架重合。为此没有复用旧源检查点，而是生成 fold1 专属净化源划分：

- 原始：2624 / 328 / 328（train/val/test）。
- 净化后：2618 / 327 / 328。
- 从源 train 排除：`1nvq, 1nvr, 3hmo, 4ogj, 5i9y, 5ia3`。
- 从源 validation 排除：`4btk`。
- 排除规则：Davis fold1 val/test 的 CSV SMILES 或 PubChem SDF，只要与 PDBbind 配体在连接性 InChIKey 主块或标准化 Murcko scaffold 任一口径重合，即从 PDBbind 的所有分区排除。

净化划分：`data/source_splits/pdbbind_fold1_clean.json`。  
审计明细：`outputs/audit/ligand_overlap_report.json` 和 `outputs/audit/ligand_overlap_details.json`。

蛋白同源性只报告、不排除：442 个 Davis 蛋白中 27 个与净化后的 PDBbind train 蛋白达到 identity ≥30%、query/target coverage 均 ≥80%。当前任务是 drug-cold，Davis train/validation/test 本来共享同一蛋白面板，因此这不是 heldout 轴泄漏。

## 新数据预处理

### 1. Davis 97维配体图

- 输入：`/data1/ztx/MyModel-MDTA/data/raw/davis/pubchem_sdf/`（68 个 SDF）。
- 方法：直接调用源实验 `experiments/pdbbind_residue_transfer/rich_ligand_features.py`；RDKit sanitize 后移除显式氢，按原 SDF 重原子顺序生成 97维原子特征、6维键特征和双向边。
- 原因：PDBbind 配体编码器只对完全相同的特征语义和原子顺序可迁移。
- 输出：`data/davis_ligand_rich97/`。
- 核验：68/68 成功；与现有 `drug_atom_features_v2` 的重原子数全部一致，逐原子坐标最大绝对差为 0；CSV SMILES 重原子数也 68/68 一致。

### 2. Davis ProtT5-1024 残基缓存

- 输入：`/data1/ztx/MyModel-MDTA/data/raw/davis/proteins.csv`（442 个蛋白 ID）。
- 模型：`/data1/ztx/DTBind/tools/prot_t5_xl_uniref50`，与 PDBbind 残基输入一致。
- 标准化：大写；`U/Z/O/B` 和其他非标准字符转 `X`；每个残基一个 token。
- 长序列：窗口 900、重叠 128；重叠区使用线性边缘权重加权平均，避免截断和硬拼接。最长序列 2549 aa。
- 去重：按标准化序列 SHA-256 缓存；442 个 ID 对应 379 条唯一序列。manifest 仍逐 ID 保存，序列不同的突变体不会合并。
- 输出：`data/davis_prott5_1024/`。
- 核验：442/442 成功，0 失败，所有张量严格为 `[sequence_length, 1024]`。

## PDBbind 源模型

旧二元 warm-start 检查点本身见过被排除的复合物，因此也没有复用，而是在净化划分上从头训练：

- 二元模型按 pair macro AUPRC 选模，最佳源 validation 为 0.084394（epoch 12）。
- 二元源 test：micro 0.028863，macro 0.061587；旧二元源 test 为 0.024239 / 0.057863。
- rich/typed 辅助模型改为按共享 `base` pair macro AUPRC 选模，而非 typed masked 指标。
- rich 最佳源 validation base macro 为 0.097554（epoch 13）。
- rich 源 test base：micro 0.031655，macro 0.074806；positive rate 0.001071，约 29.55× 随机基线。
- 最终检查点：`outputs/source_rich_base_selected_fold1_clean/source_base_selected_best.pt`。

这说明 PDBbind 分支本身的 atom–residue 学习较旧版本更准确，Davis 失败不能简单归因于源模型完全没学到东西。

## 冻结局部表示与可学习性

对每个 Davis 药物–蛋白对：

1. 用净化后的冻结 PDBbind 模型生成 residue/atom 128维表示。
2. 计算完整 base pair map。
3. 取 top-64 pair，以 `softmax(logit / 0.5)` 加权汇聚 `h_residue * h_atom`，得到 128维局部向量。
4. 同时缓存随机同构模型、原子特征打乱和错配药物三个对照。

输出：`data/local_features/fold1.pt`（30,056×128，四个条件）。

可学习性审计：

- P0 每维跨样本平均 std：1.7491；样本范数 std：12.023。
- P0 pair-map 平均 logit std：3.0432；top-64 logit std：1.1100。
- R0 pair-map 平均 logit std：0.9732。
- P0 与 atom-shuffle 局部向量平均 L2 差：25.5475。

因此局部表示不是此前 E2 中接近常数的退化特征；它有充分变化且对原子扰动敏感。

## Davis 训练协议

- 全局模型：现有 fold1 baseline 检查点完全冻结。
- 全局预测按 68 个唯一药物和 442 个唯一蛋白各编码一次后组合；与普通逐对前向抽样最大绝对差 `4.768e-7`。
- 局部模型和 PDBbind 全部冻结，只训练 bias-free 残差头：`Linear(128,32,bias=False) → SiLU → Linear(32,1,bias=False)`；末层零初始化。
- 局部向量仅用 train 均值/标准差归一化。
- AdamW，LR 1e-3，weight decay 1e-4，batch 64，最多 100 epoch，patience 15。
- seeds：42、43、44；按 validation MSE 选模。
- test 指标未计算。

## 失败原因判断

1. **不是表示坍缩。** P0 的 pair map 和 pooled vector 都有很大方差，残差也不是常数。
2. **不是源任务完全没学会。** 净化后的 rich PDBbind test base macro AUPRC 达 0.07481，并对原子打乱敏感。
3. **主要是迁移语义不对齐。** PDBbind 学的是“给定真实复合物中，哪些原子–残基形成接触”；Davis 没有配体姿态或真实口袋，当前做法在整条蛋白上直接取最高的 64 个兼容 pair。这些最高分更像源域接触先验，不等价于该药物在该蛋白上的亲和力残差。
4. **随机局部略好说明残差头可利用实体特征，但不是接触迁移收益。** 随机编码仍保留药物图和蛋白序列的随机非线性投影，相当于一个受限的实体特征补充；它的微小改善不能证明细粒度交互。
5. **atom-shuffle 更差，但不足以救 P0。** 打乱后 MSE 从 P0 的 0.49252 恶化到 0.49707，表明 P0 确实含有化学顺序信息；然而 P0 本身仍差于基线，因此该信息当前没有正确对齐 Davis 目标。
6. **mismatch 与 P0 接近，进一步否定配对特异性。** 错配配体 MSE 0.49194，与 P0 0.49252 相近，说明当前 top-k pooling 没有形成足够强的“正确药物–正确蛋白”特异性。

## 下一步建议

不要直接五折、不要直接解冻。下一步应先做一个更小的接口诊断：固定同一蛋白，对正确药物与错配药物的 top-k 残基位置做一致性/区分度分析，并把全蛋白 top-k 改成由结构或序列证据限定的候选口袋后再重复 P0 vs mismatch。只有当正确配对在候选口袋内表现出显著高于错配的配对特异性时，才值得进入 affinity 训练。若限定口袋后仍无配对特异性，应停止直接迁移 contact logits，转而用 PDBbind 做辅助多任务正则或对比对齐，而不是作为 Davis 的冻结局部分支。

## 主要文件

- 脚本：`scripts/`
- 预处理缓存：`data/davis_ligand_rich97/`、`data/davis_prott5_1024/`
- 净化源划分：`data/source_splits/pdbbind_fold1_clean.json`
- 源检查点与指标：`outputs/source_binary_fold1_clean/`、`outputs/source_rich_base_selected_fold1_clean/`
- 全局/局部缓存：`data/global_predictions/fold1.pt`、`data/local_features/fold1.pt`
- 最终对照结果：`outputs/pilot_fold1/pilot_results.json`
- 日志：`logs/`

## 后续实验：P2Rank top3 口袋约束

在全蛋白 P0 得到 NO-GO 后，继续执行了预先建议的候选口袋诊断。该步骤没有重新生成蛋白结构或重新运行 P2Rank，而是复用现有缓存：

- 输入口袋缓存：`data/processed/davis/multiscale/protein_pockets_p2rank_top3_v2.pt`。
- 选择原因：442/442 蛋白有效，平均 2.99 个口袋、18.80 residues/pocket，平均映射覆盖 99.92%；比平均约 24 个子口袋的 CAVIAR 更适合作为最小候选区约束。
- 方法：按 P2Rank score 取 top3 pocket，将 `sequence_indices` 去重合并；每个蛋白候选残基 8–96 个，平均 53.74 个。其余 PDBbind 模型、top-64、temperature=0.5 和三种子残差协议保持不变。
- 新缓存：`data/local_features_p2rank_top3/fold1.pt`，没有覆盖全蛋白缓存。

### 零训练配对特异性

使用归一化 pair-map log-mean-exp score 与 Davis affinity 做相关分析，只访问 fold1 train/validation：

| validation 指标 | PDBbind P0 | random | atom-shuffle | mismatch |
|---|---:|---:|---:|---:|
| overall Spearman | -0.0341 | 0.1166 | -0.0421 | 0.0010 |
| 同一蛋白内跨 7 个 heldout drugs 平均 Spearman | 0.0292 | 0.2155 | 0.0245 | 0.0210 |
| 同一 drug 跨 proteins 平均 Spearman | -0.0593 | 0.0644 | -0.0633 | -0.0162 |

P0 score 对 Davis affinity 没有稳定正相关，说明口袋约束没有把接触置信度直接变成亲和力排序信号。

### 三种子残差结果

| 条件 | MSE mean ± std | 改善种子数 |
|---|---:|---:|
| G0 冻结全局基线 | 0.489465 | — |
| G1 affine | 0.490584 ± 0.000628 | 0/3 |
| R0 pocket-random | 0.491586 ± 0.001477 | 0/3 |
| P0 pocket-PDBbind | 0.490632 ± 0.000822 | 0/3 |
| P-shuffle | 0.494242 ± 0.000980 | 0/3 |
| P-mismatch | 0.494268 ± 0.001523 | 0/3 |

相对全蛋白 P0（0.492517），P2Rank 约束把 P0 改善到 0.490632，并使 shuffle/mismatch 明显更差，说明候选口袋确实减少了部分伪高分并增强了配对约束。然而 P0 仍比全局基线恶化 0.238%，0/3 种子改善，go/no-go 仍为 **NO-GO**。Davis test 仍未访问。

### 更新后的路线判断

不再继续调 top-k、temperature、残差头宽度或换 CAVIAR 口袋；这些只能微调同一个未对齐信号。下一步若继续，应改变迁移目标：

1. 不把 PDBbind contact logits 当作 Davis affinity 的直接输入；改为在 Davis 训练时保留一个 PDBbind contact-preservation 辅助损失，约束表示不要遗忘局部化学知识。
2. 引入“正确 drug–protein 相对错配”的跨域对比对齐，让正确组合的 pocket-local 表示相对错配可区分；目前 PDBbind 单独预训练没有获得这种 Davis 配对语义。
3. 在任何 affinity 主实验前，先要求 validation 的同蛋白 drug-ranking Spearman 明显高于 random/mismatch；否则停止该版本，不进入五折。

新增输出：

- `data/local_features_p2rank_top3/fold1.json`
- `outputs/pocket_specificity/fold1.json`
- `outputs/pilot_fold1_p2rank_top3/pilot_results.json`

## 后续实验：受保护的 PDBbind contact 辅助对齐

本轮不再把冻结 contact 表示直接输入 Davis affinity 回归。主路径保持原
`p13d_earlystop` 简单拼接模型不变，PDBbind 只作为训练期教师：

1. 从原模型回归头之前缓存精确的 256维 drug–protein 拼接表示；预测与普通逐对前向的最大绝对差仍为 `4.768e-7`。
2. 学生适配器为 `Linear(256,128) → SiLU → LayerNorm`，用 cosine loss 对齐 PDBbind P2Rank-top3 contact 表示。
3. 学生后接零初始化 bias-free residual head；最终预测始终为 `原 p13d 预测 + residual`。
4. PDBbind 教师只参与 train loss，validation 推理及将来部署均不需要 PDBbind 特征。
5. epoch 0 明确定义为“完全关闭辅助分支”的原始基线；只按 validation affinity MSE 选模，且至少改善 `1e-5` 才启用辅助分支，否则保存 `enabled=false` 并逐元素返回原始预测。

新派生缓存（不是新的原始数据预处理）：

- `data/global_predictions/fold1_with_features.pt`：30,056×256 冻结拼接表示、原预测、标签和行 ID。
- 生成原因：让学生只读取原主任务已经拥有的表示，同时保证 PDBbind 教师不会进入推理输入。
- 数值核验：30,056 行全部有限；68 个 drug、442 个 protein；未计算 test 指标。

协议：fold1 train 20,774 行、validation 3,094 行；3 seeds；AdamW，LR `1e-3`，weight decay `1e-4`，batch 512，最多 100 epoch，patience 15，contact alignment weight `0.05`。测试集未访问。

### 保护后结果

| 条件 | 最终 validation MSE mean ± std | 启用种子数 |
|---|---:|---:|
| 原冻结 p13d | 0.489465 ± 0 | — |
| 仅学生残差、无教师 | 0.489465 ± 0 | 0/3 |
| random teacher | 0.489465 ± 0 | 0/3 |
| PDBbind contact teacher | 0.489465 ± 0 | 0/3 |
| atom-shuffle teacher | 0.489465 ± 0 | 0/3 |
| mismatch teacher | 0.489465 ± 0 | 0/3 |

这不是训练没有发生。PDBbind 三个种子的最佳未保护 validation MSE 为
`0.499955 / 0.498621 / 0.496531`，都差于基线；其 alignment loss 从约
`0.87` 降至约 `0.807`，train affinity MSE 从约 `0.219–0.221` 降至
`0.183` 左右。因此学生确实吸收了 PDBbind 表示，但增强的是训练集拟合，
没有形成对 heldout drugs 的泛化增益。random、shuffle、mismatch 也出现同样的
train 降低而 validation 变差现象。

结论仍为 **NO-GO**，但达成了设计目标：PDBbind 信息若无验证增益便被自动拒绝，
原 `p13d_earlystop` 主任务不受影响。本轮不应扩展到五折，也不应仅靠调大
alignment weight 或放宽保护阈值来强行启用。若继续探索，应改变监督层级，优先
考虑在 PDBbind 阶段直接学习可迁移的 pair-level 对比目标，或在 Davis 上获得
独立于 affinity validation 的局部伪标签；当前 pair-level contact embedding 的
表示蒸馏仍缺少 Davis drug-cold 所需的可泛化配对语义。

### 50 epoch train-only 预蒸馏强化核验

为排除前述联合训练因 patience 15 而使教师尚未充分蒸馏的可能，又在独立输出目录
运行了两阶段版本：先仅用 fold1 train 行做 50 epoch contact 表示蒸馏，再重新以
零残差开始受保护 affinity 训练。validation 不参与预蒸馏，test 仍未访问。

- PDBbind teacher 的 train cosine loss 从 `0.835–0.844` 降至 `0.508–0.510`，
  证明学生已明显吸收教师表示。
- PDBbind 三个种子的最佳未保护 validation MSE 分别为
  `0.497176 / 0.500575 / 0.500005`，仍全部差于基线，最终均回退到基线（0/3 enabled）。
- random teacher 有且仅有 seed 43 改善至 `0.486962`（约 +0.511%），另两个种子
  失败；保护后的三种子均值为 `0.488631 ± 0.001445`。真实 PDBbind teacher 为
  `0.489465 ± 0`，0/3 enabled，因此随机波动不能解释为 contact 知识收益。
- atom-shuffle 与 mismatch 均为 0/3 enabled；所有条件的受保护最终结果均不差于基线。

该核验排除了“仅仅是蒸馏不足”。结论是：当前冻结 p13d 拼接表示能够预测相当一部分
contact teacher 表示，但这部分信息没有提升 Davis drug-cold affinity 泛化。

新增脚本与结果：

- `cache_global_predictions.py`
- `train_auxiliary_contact_distillation.py`
- `outputs/auxiliary_contact_distillation_fold1/auxiliary_results.json`
- `outputs/auxiliary_contact_distillation_pretrain50_fold1/auxiliary_results.json`
