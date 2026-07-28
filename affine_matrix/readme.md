solve_bus_affine_g.m计算仿射再调度矩阵G  
cluster_buses.m对118个母线进行K-means聚类  
dfs_refine_pima_clusters.m深度优先搜索将一组中不连通的母线分到其他组中  
partition_pima_case118.m运行入口  
数据链路：pima_affine_solution.mat→ K-means 重新生成 bus_clusters.csv→ DFS 连通修复→ 拓扑均衡 MIP→case118_clusters.json  
