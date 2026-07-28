function result = partition_pima_case118(outputJson)

arguments
    outputJson (1,1) string = ""
end

scriptDir = fileparts(mfilename('fullpath'));
repoDir = fileparts(scriptDir);
affineFile = fullfile(scriptDir,'pima_affine_solution.mat');
if outputJson == ""
    outputJson = fullfile(repoDir,'COPF','case118_clusters.json');
end

initial = cluster_buses( ...
    input_file=affineFile,output_dir=scriptDir,k=3,k_range=3, ...
    replicates=100,make_plots=false);

refined = dfs_refine_pima_clusters( ...
    case_name="case118",affine_file=affineFile, ...
    cluster_file=fullfile(scriptDir,'bus_clusters.csv'), ...
    output_dir=scriptDir,auto_refine=true,balance=true, ...
    feature_weight=1,cut_weight=10,json_file=outputJson);

result.initial = initial;
result.refined = refined;
result.output_json = char(outputJson);
end
