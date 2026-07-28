运行命令：python multi-dnn model/main.py --data case118_n1_per_branch_N140.mat --training-mode partitioned --contingency onehot --epochs 200 --device cuda --out-dir runs/case118_partitioned  
--case 可选case118和case300  
使用不同算例或分组方式要提供对应的--cluster-file，分组配置入口：COPF/case118_clusters.json  
--date 提供训练数据.mat文件（当前训练数据case118_n1_per_branch_N140.mat包含26,040 条样本（20,832条训练；5,208条测试）186 种支路 N-1 故障）  
--training-mode 可选单DNN(single)或分组多DNN(partitioned)  
--contingency 选择N-1故障输入编码方式，scalar每个DNN只增加一个输入维度，onehot使用186维故障向量   
--validation-ratio 从.mat已有的train_idx再划出多少比例用于验证(默认0.2)  
--selection-warmup 从最开始多少个epoch才开始选最佳模型(默认50)  
--loss-candidate 设置不同惩罚系数，依次传入五个参数cy, cv, cp, cq, rho，（默认1,1,1,1,0.001），前四个用于修改惩罚力度，rho是电压、Pg/Qg 上下界违反的拉格朗日乘子更新速度  
其中cy是真实标签 Vm/Va/Pg/Qg 的拟合损失，cv虚拟节点 Vm/Va 一致性损失，cp是支路有功潮流误差惩罚，cq是支路无功潮流误差惩罚  

用单个连续数表示 186 种线路故障，可能会给网络引入“相邻线路编号具有相似含义”的错误归纳偏置。将scalar改成onehot之后各指标变化：

| 指标 | Scalar | One-hot | 变化 |
|------|--------|---------|------|
| Normalized MSE | 1.0118 | 0.1146 | 降低 88.7% |
| 支路有功 RMSE | 19.18 MW | 6.04 MW | 降低 68.5% |
| 支路无功 RMSE | 6.49 Mvar | 2.21 Mvar | 降低 66.0% |
| 电压 RMSE | 0.00958 pu | 0.00214 pu | 降低 77.7% |
| 相角 RMSE | 2.31° | 1.61° | 降低 30.2% |
| Pg RMSE | 13.41 MW | 3.33 MW | 降低 75.2% |
| Qg RMSE | 24.53 Mvar | 2.61 Mvar | 降低 89.3% |
| 平均绝对成本 gap | 2.105% | 1.302% | 降低 38.1% |
| 最大绝对成本 gap | 21.35% | 6.48% | 降低 69.6% |


复现论文方法的关键机制：三分区、多阶段DNN、虚拟节点、流损失和拉格朗日式约束更新；而且带符号平均最优性差距达到论文 <0.1% 的量级，但约束可行率与论文 M7 仍有明显差距
| 指标 | 结果 |
|------|------|
| ΔP | 0.145269 MW |
| ΔQ | 0.118955 Mvar |
| Δθ | 0.071941° |
| ΔV | 0.015370 kV |
| ηV | 98.593% |
| ηPg | 91.702% |
| ηQg | 91.514% |
| 带符号平均最优性差距 | 0.0643% |

可能因为各分区输入只包含本区节点的有功、无功负荷以及全网线路故障编码，无法直接获知其他分区的负荷变化和全网总负荷，因此难以协调跨区发电机的 Pg、Qg调节。目前三个分区独立训练并在推理阶段直接拼接，局部预测误差可能在全网组合时累积。此外，当前可行性判断采用 (10^{-4}) p.u. 的严格容差，而模型输出没有使用有界映射或推理后投影，轻微越界也会被判定为不满足约束。
