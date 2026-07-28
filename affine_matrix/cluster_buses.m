function result = cluster_buses(o)
%CLUSTER_BUSES Cluster buses from the affine state-response matrix.

arguments
    o.input_file (1,1) string = ""
    o.output_dir (1,1) string = ""
    o.k (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    o.k_range (1,:) double {mustBeInteger,mustBePositive} = 2:12
    o.replicates (1,1) double {mustBeInteger,mustBePositive} = 50
    o.max_iter (1,1) double {mustBeInteger,mustBePositive} = 2000
    o.seed (1,1) double {mustBeInteger,mustBeNonnegative} = 42
    o.noise_relative (1,1) double {mustBeNonnegative,mustBeFinite} = 1e-8
    o.make_plots (1,1) logical = true
end

assert(exist('kmeans','file') == 2, ...
    'Statistics and Machine Learning Toolbox is required for kmeans.');

scriptDir = fileparts(mfilename('fullpath'));
if o.input_file == ""
    inputFile = fullfile(scriptDir,'pima_affine_solution.mat');
else
    inputFile = char(o.input_file);
end
if o.output_dir == ""
    outputDir = scriptDir;
else
    outputDir = char(o.output_dir);
end
assert(isfile(inputFile), 'Input file does not exist: %s', inputFile);
if ~isfolder(outputDir), mkdir(outputDir); end

%% Load the published state matrix and bus identifiers
[~,~,inputExt] = fileparts(inputFile);
if strcmpi(inputExt,'.mat')
    data = load(inputFile);
    assert(isfield(data,'r') && isfield(data.r,'G_state'), ...
        'The MAT file must contain r.G_state.');
    Gstate = double(data.r.G_state);
    if isfield(data.r,'bus_ids')
        busIds = double(data.r.bus_ids(:));
    else
        busIds = (1:(size(Gstate,1)/2))';
    end
else
    Gstate = readmatrix(inputFile);
    busIds = (1:(size(Gstate,1)/2))';
end

assert(all(isfinite(Gstate),'all'), 'G_state contains NaN or Inf values.');
assert(mod(size(Gstate,1),2) == 0, ...
    'G_state must have two rows per bus: [V_i; theta_i].');
nb = size(Gstate,1)/2;
assert(numel(busIds) == nb, 'The number of bus IDs does not match G_state.');

% G_state is normally [-Astate,Astate]. Keeping only +Astate removes exact
% duplicate information without changing any pairwise K-means distances.
if size(Gstate,2) == 2*nb
    negationError = max(abs(Gstate(:,1:nb)+Gstate(:,nb+1:end)),[],'all');
    negationScale = max(1,max(abs(Gstate),[],'all'));
    assert(negationError <= 1e-8*negationScale, ...
        'Expected G_state(:,1:N) = -G_state(:,N+1:2N).');
    Astate = Gstate(:,nb+1:end);
elseif size(Gstate,2) == nb
    negationError = NaN;
    Astate = Gstate;
else
    error('G_state must have N or 2N columns for N buses.');
end

%% Form one feature vector per bus
Vraw = Astate(1:2:end,:);
ThetaRaw = Astate(2:2:end,:);

% Remove only solver-level noise before scaling. Column-wise z-scoring is
% intentionally avoided because it would amplify zero-injection columns.
vTol = o.noise_relative*max(abs(Vraw),[],'all');
thetaTol = o.noise_relative*max(abs(ThetaRaw),[],'all');
Vraw(abs(Vraw) < vTol) = 0;
ThetaRaw(abs(ThetaRaw) < thetaTol) = 0;

vScale = norm(Vraw,'fro')/sqrt(numel(Vraw));
thetaScale = norm(ThetaRaw,'fro')/sqrt(numel(ThetaRaw));
assert(vScale > 0 && thetaScale > 0, ...
    'Voltage or angle response block is identically zero.');
Vfeature = Vraw/vScale;
ThetaFeature = ThetaRaw/thetaScale;
X = [Vfeature,ThetaFeature];

%% Evaluate candidate K values
kCandidates = unique([o.k_range(:);o.k]);
kCandidates = kCandidates(kCandidates >= 2 & kCandidates < nb);
if o.k == 0
    assert(numel(kCandidates) >= 3, ...
        'Automatic elbow selection requires at least three K candidates.');
else
    assert(any(kCandidates == o.k), 'k must satisfy 2 <= k < number of buses.');
end

nCandidates = numel(kCandidates);
wcss = zeros(nCandidates,1);
meanSilhouette = zeros(nCandidates,1);
labelsByK = cell(nCandidates,1);
centersByK = cell(nCandidates,1);

for j = 1:nCandidates
    k = kCandidates(j);
    rng(o.seed+k,'twister');
    [labels,centers,sumd] = kmeans(X,k, ...
        'Distance','sqeuclidean', ...
        'Start','plus', ...
        'Replicates',o.replicates, ...
        'MaxIter',o.max_iter, ...
        'Display','off');
    wcss(j) = sum(sumd);
    silhouetteValues = silhouette(X,labels,'sqeuclidean');
    silhouetteValues = silhouetteValues(isfinite(silhouetteValues));
    meanSilhouette(j) = mean(silhouetteValues);
    labelsByK{j} = labels;
    centersByK{j} = centers;
end

% The elbow is the greatest downward departure of log(WCSS) from the line
% joining the first and last candidates. Endpoints cannot be the elbow.
logWcss = log(max(wcss,realmin));
xNorm = (kCandidates-kCandidates(1))/(kCandidates(end)-kCandidates(1));
if abs(logWcss(1)-logWcss(end)) > eps
    yNorm = (logWcss-logWcss(end))/(logWcss(1)-logWcss(end));
    elbowScore = (1-xNorm)-yNorm;
else
    elbowScore = zeros(size(logWcss));
end
elbowScore([1,end]) = -Inf;

if o.k == 0
    [~,selectedPosition] = max(elbowScore);
    selectedK = kCandidates(selectedPosition);
    selectionMethod = 'log-WCSS elbow';
else
    selectedK = o.k;
    selectedPosition = find(kCandidates == selectedK,1);
    selectionMethod = 'user supplied';
end

labels = labelsByK{selectedPosition};
centers = centersByK{selectedPosition};

clusterMinBus = accumarray(labels,busIds,[selectedK,1],@min,Inf);
[~,clusterOrder] = sort(clusterMinBus);
labelMap = zeros(selectedK,1);
labelMap(clusterOrder) = 1:selectedK;
labels = labelMap(labels);
centers = centers(clusterOrder,:);

%% Save results
responseNorm = sqrt(sum(X.^2,2));
clusterTable = table(busIds,labels,responseNorm, ...
    'VariableNames',{'bus_id','cluster','normalized_response_norm'});
finiteElbowScore = elbowScore;
finiteElbowScore(~isfinite(finiteElbowScore)) = NaN;
kTable = table(kCandidates,wcss,meanSilhouette,finiteElbowScore, ...
    'VariableNames',{'k','within_cluster_sum_squares', ...
    'mean_silhouette','elbow_score'});

writetable(clusterTable,fullfile(outputDir,'bus_clusters.csv'));
writetable(kTable,fullfile(outputDir,'k_selection.csv'));
writematrix(X,fullfile(outputDir,'cluster_features.csv'));
writematrix(centers,fullfile(outputDir,'cluster_centers.csv'));

result.input_file = inputFile;
result.bus_ids = busIds;
result.cluster = labels;
result.selected_k = selectedK;
result.selection_method = selectionMethod;
result.k_candidates = kCandidates;
result.wcss = wcss;
result.mean_silhouette = meanSilhouette;
result.elbow_score = finiteElbowScore;
result.features = X;
result.centers = centers;
result.voltage_scale = vScale;
result.angle_scale = thetaScale;
result.voltage_noise_threshold = vTol;
result.angle_noise_threshold = thetaTol;
result.negation_error = negationError;
result.options = o;
save(fullfile(outputDir,'cluster_result.mat'),'result');

fprintf('Loaded %d buses from %s\n',nb,inputFile);
fprintf('Selected K = %d by %s; mean silhouette = %.4f\n', ...
    selectedK,selectionMethod,meanSilhouette(selectedPosition));
for c = 1:selectedK
    members = busIds(labels == c)';
    fprintf('Cluster %d (%d buses): %s\n',c,numel(members), ...
        strjoin(string(members),', '));
end
fprintf('Results written to %s\n',outputDir);
end
