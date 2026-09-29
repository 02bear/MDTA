# P13D additive bilinear rank16 两折开发实验

固定 Fold1 / Fold3，使用 v2_final split。从头训练，seed42。两折均完成后比较，不按中途验证结果调整结构或参数。

保持原 drug1d、drug3d三层EGNN、protein1d、protein3d三层EGNN及各模态融合模块；替换最终 concat MLP decoder 为 a(d)+b(p)+dot(Ud,Vp)/sqrt(16)。U/V维度128→16，无偏置，随机初始化，双侧第一步均可获得梯度。a/b为线性主效应，只有a含全局bias。融合向量输入处dropout0.1。

这是预测头整体替换，参数量也不同，不宣称参数量匹配或只添加一个等价模块。原decoder有33025参数，新头4353参数。编码器初始权重在相同seed下逐张量一致，已在真实模型smoke test核查。

其余命令参数逐项取自各折 baseline checkpoint 的args；不把checkpoint权重载入实验模型。batch16、lr3e-4、weight_decay1e-5、Adam、MSE、max500epoch、patience60、min_delta1e-4、num_workers0。沿用原数据读取/训练/earlystop代码；增加只读验证诊断和最佳预测保存。无新数据预处理，不读outer test标签做评估。

诊断：每epoch保存预测交互RMS、label≥7 MSE、逐药物MSE（按药物ID排序）；best_val_predictions.npz包含完整ID、索引、预测与标签。原MSE/CI/Rm2评估代码不变。

开发成功口径：两折最佳验证MSE分别优于0.7051149011和0.7712797523；CI/Rm2、困难药物和强结合子组作为一致性检查。交互RMS非零本身不是成功。两折已用于诊断，不能称独立确认。

GPU0运行Fold1，GPU1运行Fold3；worker在该卡可用显存≥34000MiB时启动，否则每60秒检查一次。训练失败标记failed，不自动改参数或跳过。来源代码与split SHA256在locked_protocol.json锁定；排队结束再次核验。

代码目录：/data1/ztx/MyModel-MDTA/experiments/similarity_balanced_drugcold_5fold/bilinear_trial_20260907/

输出目录：/data1/ztx/MyModel-MDTA/outputs/Refine_experiment/davis/cold_start/drug_cold_similarity_affinity_balanced_5fold_seed42_v2_final/bilinear_rank16_trial_20260907/
