**solve_bus_affine_g.m**计算仿射再调度矩阵G  
与论文的实现还存在以下区别：  
1.式(11)联合优化基准变量 z 和仿射策略 G；代码先求出并固定基准 AC-OPF，只优化响应矩阵。  
2.论文用线路故障的导纳变化与注入变化构造不确定性；代码不改变线路导纳或拓扑，只施加母线净注入扰动。  
3.式(15) 的列数 2|N^U| 表示每个不确定母线的两类注入分量；代码假设扰动±15%，P/Q 完全相关，随后用负、正两个端点列拼出相同数量的列。  

**cluster_buses.m**对118个母线进行K-means聚类，  
result = cluster_pima_buses/
result = cluster_pima_buses(k=6)指定组数为6/  
result = cluster_pima_buses(k=0, k_range=2:15, replicates=100);应用肘部法则确定组数  

**dfs_refine_pima_clusters.m**深度优先搜索将一组中不连通的母线分到其他组中  
**partition_pima_case118.m**运行入口  

数据链路：pima_affine_solution.mat→ K-means 重新生成 bus_clusters.csv→ DFS 连通修复→ 拓扑均衡 MIP→case118_clusters.json  
