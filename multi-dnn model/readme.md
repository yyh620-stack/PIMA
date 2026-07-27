运行命令：python COPF/main.py --data COPF/case118_n1_per_branch_N140.mat --training-mode partitioned --model two-stage --contingency scalar--epochs 200 --device cuda --out-dir runs/case118_partitioned  
--case 可选case118和case300  
使用不同算例或分组方式要提供对应的--cluster-file，分组配置入口：COPF/case118_clusters.json  
--date 提供训练数据.mat文件（当前训练数据case118_n1_per_branch_N140.mat包含26,040 条样本（20,832条训练；5,208条测试）186 种支路 N-1 故障）  
--training-mode 可选单DNN(single)或分组多DNN(partitioned)  
--contingency 选择N-1故障输入编码方式，scalar每个DNN只增加一个输入维度，onehot使用186维故障向量  
--cp 
--rho 拉格朗日乘子的更新步长  
--validation-ratio 从.mat已有的train_idx再划出多少比例用于验证(默认0.2)  
--selection-warmup 从最开始多少个epoch才开始选最佳模型(默认50)  
