# BRICS 层级药物编码 Fold1 验证报告

## 最终结论

BRICS 信息可以为 P13D 的 Davis 药物冷启动任务提供增益，但有效成分不是“片段表示回写到对应 P13D 原子”，而是：

1. 显式的片段化学属性；
2. 真实 BRICS 片段连接图；
3. 在 affinity 训练前仅用训练药物进行 masked-fragment chemistry 预训练。

当前不应解冻 atom-to-fragment 回写路径。下一版应删除该路径，只保留预训练 BRICS 化学图作为独立药物结构视角，在 P13D 药物融合层进行受控融合。

所有实验只使用 train/validation，未物化、计算或选择 test split。

## 新数据预处理

### 1. 轻量 BRICS 映射

- 输入：`experiments/klifs85_dual_graph/data/fold1/entity_graphs.pt`
- 输出：`experiments/brics_hierarchical/data/brics_mappings.pt`
- 内容：每个药物的显式氢原子到 BRICS 片段编号、片段边、7 维断裂键边属性。
- 统计：68 个药物；35–83 个原子；3–13 个片段，均值 7.5882。

### 2. P13D 冻结药物编码缓存

- 输出：`experiments/brics_hierarchical/data/fold1_drug_encoder_inputs.pt`
- 内容：Drug3DEGNN 原子节点、Drug3DEGNN 全局表示、Drug1D 表示、原始药物融合表示及 BRICS 映射。
- checkpoint：fold1 P13D baseline epoch 104。
- 重新计算的药物融合表示与原 P13D pair cache 最大误差：`9.54e-7`。

### 3. 片段显式化学特征

- 输出：`experiments/brics_hierarchical/data/fold1_drug_encoder_inputs_chem.pt`
- 516 个 BRICS 片段，每个片段 282 维：
  - 256 维、radius 2 的 fragment-centered Morgan fingerprint；
  - 16 维 BRICS 断点环境；
  - 10 维标准化原子组成、质量、芳香性、环及电荷描述符。
- 以已经审计的 `atom_to_fragment` 为唯一片段边界，没有重新生成或改写映射。

## 模型

1. 冻结 P13D Drug3DEGNN 产生原子特征。
2. 按 BRICS 聚合原子 mean/max，并拼接 282 维片段化学特征。
3. 两层 GATv2 在 BRICS 片段图上传播。
4. masked-fragment 任务恢复片段指纹、断点环境和描述符。
5. 片段表示广播到原子，生成 Drug3D 表示增量；增量输出零初始化，epoch 0 精确等于 P13D。
6. Drug1D、Protein 表示、P13D fusion/decoder 均保持冻结。

## 直接联合训练

P13D 基线：MSE 0.489465，CI 0.788562，Rm² 0.341028。

| 条件 | seed | best epoch | MSE | 相对基线 | CI | Rm² |
|---|---:|---:|---:|---:|---:|---:|
| real BRICS chemistry | 42 | 88 | 0.466177 | +4.76% | 0.796884 | 0.371494 |
| random atom assignment | 42 | 23 | 0.480394 | +1.85% | 0.800828 | 0.358725 |
| no fragment edges | 42 | 23 | 0.481645 | +1.60% | 0.795815 | 0.356143 |

直接联合训练在 seed42 有较强收益，但 seed43 仅改善 0.70%，seed44 回退 epoch 0。原因是片段化学恢复需要较长训练，而 affinity early stopping 可能在表示成熟前终止。

## 训练药物化学预训练 → affinity

预训练严格只使用 fold1 的 47 个训练药物，每个种子 3000 步；不使用验证药物或测试药物的标签/样本。化学恢复损失由约 0.42 降至约 0.07，再进入 affinity early stopping。

| real 条件 | best epoch | MSE | 相对基线 | MAE | CI | Rm² | 改善验证药物 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed42 | 30 | 0.476311 | +2.69% | 0.406796 | 0.792798 | 0.359614 | 5/7 |
| seed43 | 27 | 0.480438 | +1.84% | 0.406008 | 0.799203 | 0.356057 | 5/7 |
| seed44 | 90 | 0.466614 | +4.67% | 0.404980 | 0.791681 | 0.371107 | 5/7 |
| 三种子均值 | — | 0.474454 | +3.07% | 0.405928 | 0.794561 | 0.362259 | — |

MSE 跨种子标准差为 0.00710。每个种子的药物级 bootstrap 正收益概率约 0.72–0.87，但由于验证集只有 7 个冷启动药物，单种子的 95% 区间仍跨 0。

## 匹配因果对照：预训练 → affinity，seed42

| 条件 | MSE | 相对基线 | CI | Rm² |
|---|---:|---:|---:|---:|
| real | 0.476311 | +2.69% | 0.792798 | 0.359614 |
| random atom assignment | 0.476118 | +2.73% | 0.798650 | 0.362205 |
| no fragment edges | 0.485813 | +0.75% | 0.791942 | 0.348058 |

### 因果解释

- random atom assignment 与 real 几乎相同，并略优 0.000192 MSE：没有证据支持正确 atom-to-fragment 回写带来收益。
- 删除片段边后明显退化约 0.00950 MSE：真实 BRICS 片段连接图具有增量价值。
- random assignment 仍保留真实片段化学节点和真实片段边。因此其收益说明可迁移信号来自 BRICS 化学图，而不是 P13D 原子与片段的精确对齐。

## 决策

1. 不进入当前 atom-writeback 模型的全阶段解冻。
2. 下一模型删除原子回写和原子门控，仅保留：`BRICS chemistry node -> fragment GAT -> graph pooling`。
3. 将 BRICS graph embedding 作为独立药物结构视角，在 P13D drug fusion 前后采用零初始化、限幅门控增量。
4. 保留训练药物 masked-chemistry 预训练，再做三个种子和 no-edge 对照。
5. 只有简化模型稳定通过后，才低学习率解冻 Drug3DEGNN 第三层；KLIFS 分支随后按同样原则单独验证，暂不与 BRICS 做无监督跨图注意力。

## 文件位置

- 服务器实验根目录：`/data1/ztx/MyModel-MDTA/experiments/brics_hierarchical/`
- 本地脚本：当前目录下的 `*.py`
- 本地关键结果与审计：`results/`
