function result = dfs_refine_pima_clusters(o)
%DFS_REFINE_PIMA_CLUSTERS Make affine clusters connected and balanced.

% By default, the function reads bus_clusters.csv produced by cluster_buses. Topology is read from the selected MATPOWER case.

arguments
    o.case_name (1,1) string = "case118"
    o.affine_file (1,1) string = ""
    o.cluster_file (1,1) string = ""
    o.output_dir (1,1) string = ""
    o.auto_refine (1,1) logical = true
    o.balance (1,1) logical = true
    o.noise_relative (1,1) double {mustBeNonnegative,mustBeFinite} = 1e-8
    o.max_steps (1,1) double {mustBeInteger,mustBePositive} = 500
    o.max_balance_rounds (1,1) double {mustBeInteger,mustBePositive} = 50
    o.balance_time_limit (1,1) double {mustBePositive,mustBeFinite} = 300
    o.balance_mip_gap (1,1) double {mustBeNonnegative,mustBeFinite} = 0.01
    o.feature_weight (1,1) double {mustBeNonnegative,mustBeFinite} = 1
    o.cut_weight (1,1) double {mustBeNonnegative,mustBeFinite} = 10
    o.json_file (1,1) string = ""
end

assert(exist('loadcase','file') == 2, 'MATPOWER is required.');
scriptDir = fileparts(mfilename('fullpath'));
if o.affine_file == ""
    affineFile = fullfile(scriptDir,'pima_affine_solution.mat');
else
    affineFile = char(o.affine_file);
end
if o.output_dir == ""
    outputDir = scriptDir;
else
    outputDir = char(o.output_dir);
end
if ~isfolder(outputDir), mkdir(outputDir); end

[busIds,X] = loadAffineFeatures(affineFile,o.noise_relative);
if o.cluster_file == ""
    clusterFile = fullfile(scriptDir,'bus_clusters.csv');
else
    clusterFile = char(o.cluster_file);
    if ~isfile(clusterFile)
        clusterFile = fullfile(scriptDir,clusterFile);
    end
end
assert(isfile(clusterFile), ['Initial cluster file does not exist: %s. ', ...
    'Run cluster_pima_buses first or pass cluster_file.'],clusterFile);
initialLabels = loadClusterCsv(clusterFile,busIds);
clusterSource = clusterFile;

adjacency = caseAdjacency(char(o.case_name),busIds);
networkComponents = dfsComponents(adjacency,(1:numel(busIds))');
assert(numel(networkComponents) == 1, ...
    'The active base network itself has %d connected components.', ...
    numel(networkComponents));

fprintf('Initial connectivity from %s\n',clusterSource);
before = connectivityReport(adjacency,initialLabels,busIds);
printConnectivity(before,'BEFORE DFS REFINEMENT');
writeConnectivity(before,fullfile(outputDir,'dfs_connectivity_before.csv'));

labels = initialLabels;
moves = emptyMoveLog();
connectivitySteps = 0;
if o.auto_refine
    for step = 1:o.max_steps
        current = connectivityReport(adjacency,labels,busIds);
        if all([current.is_connected])
            break;
        end

        action = selectRepairAction(adjacency,labels,X,current,busIds);
        assert(action.valid, ...
            'No connectivity-preserving repair action was found.');
        oldLabels = labels(action.nodes);
        labels(action.nodes) = action.to_cluster;
        connectivitySteps = step;

        moves(end+1) = makeMoveRecord(step,action,oldLabels,busIds); %#ok<AGROW>
        fprintf(['Step %d: %s; buses [%s] -> Cluster %d ', ...
            '(feature delta %.6g)\n'],step,action.type, ...
            strjoin(string(busIds(action.nodes)'),', '), ...
            action.to_cluster,action.feature_delta);
    end
end

connected = connectivityReport(adjacency,labels,busIds);
if o.auto_refine
    assert(all([connected.is_connected]), ...
        'DFS refinement did not converge in %d steps.',o.max_steps);
end

balanceRounds = 0;
if o.balance
    assert(all([connected.is_connected]), ...
        'Clusters must be connected before balancing.');
    [labels,moves,balanceRounds] = balanceConnectedClusters( ...
        adjacency,labels,X,busIds,moves,o.max_balance_rounds, ...
        o.feature_weight,o.cut_weight,o.balance_time_limit, ...
        o.balance_mip_gap);
end

after = connectivityReport(adjacency,labels,busIds);
assert(all([after.is_connected]), ...
    'A cluster became disconnected during balancing.');
sizes = [after.number_of_buses];
if o.balance
    assert(max(sizes)-min(sizes) <= 1, ...
        'Balanced refinement ended with cluster sizes [%s].', ...
        strjoin(string(sizes),', '));
end
printConnectivity(after,'AFTER CONNECTED BALANCED REFINEMENT');
writeConnectivity(after,fullfile(outputDir,'dfs_connectivity_after.csv'));

changed = labels ~= initialLabels;
clusterTable = table(busIds,initialLabels,labels,changed, ...
    'VariableNames',{'bus_id','initial_cluster','refined_cluster','changed'});
writetable(clusterTable,fullfile(outputDir,'bus_clusters_dfs_refined.csv'));
writeMoves(moves,fullfile(outputDir,'dfs_moves.csv'));

result.case_name = char(o.case_name);
result.affine_file = affineFile;
result.cluster_source = clusterSource;
result.bus_ids = busIds;
result.initial_cluster = initialLabels;
result.refined_cluster = labels;
result.features = X;
result.connectivity_before = before;
result.connectivity_after = after;
result.moves = moves;
result.all_connected = all([after.is_connected]);
result.cluster_sizes = sizes;
result.connectivity_steps = connectivitySteps;
result.balance_rounds = balanceRounds;
result.options = o;
save(fullfile(outputDir,'dfs_refined_result.mat'),'result');

if o.json_file == ""
    jsonFile = fullfile(outputDir,char(o.case_name+"_clusters_balanced.json"));
else
    jsonFile = char(o.json_file);
end
writeClusterJson(labels,busIds,jsonFile);
result.json_file = jsonFile;
save(fullfile(outputDir,'dfs_refined_result.mat'),'result');

fprintf('\nRefined clusters:\n');
for c = 1:max(labels)
    members = busIds(labels == c)';
    fprintf('Cluster %d (%d buses): %s\n',c,numel(members), ...
        strjoin(string(members),', '));
end
fprintf('Changed %d of %d buses. Results written to %s\n', ...
    nnz(changed),numel(busIds),outputDir);
fprintf('Balanced JSON written to %s\n',jsonFile);
end


function [busIds,X] = loadAffineFeatures(affineFile,noiseRelative)
assert(isfile(affineFile), 'Affine result does not exist: %s',affineFile);
data = load(affineFile);
assert(isfield(data,'r') && isfield(data.r,'G_state'), ...
    'Affine MAT file must contain r.G_state.');
Gstate = double(data.r.G_state);
busIds = double(data.r.bus_ids(:));
nb = numel(busIds);
assert(isequal(size(Gstate),[2*nb,2*nb]), ...
    'Expected G_state to have size 2N-by-2N.');
negationError = max(abs(Gstate(:,1:nb)+Gstate(:,nb+1:end)),[],'all');
assert(negationError <= 1e-8*max(1,max(abs(Gstate),[],'all')), ...
    'The two halves of G_state are not exact negative counterparts.');

Astate = Gstate(:,nb+1:end);
V = Astate(1:2:end,:);
Theta = Astate(2:2:end,:);
V(abs(V) < noiseRelative*max(abs(V),[],'all')) = 0;
Theta(abs(Theta) < noiseRelative*max(abs(Theta),[],'all')) = 0;
vScale = norm(V,'fro')/sqrt(numel(V));
thetaScale = norm(Theta,'fro')/sqrt(numel(Theta));
assert(vScale > 0 && thetaScale > 0, ...
    'Voltage or angle feature block is identically zero.');
X = [V/vScale,Theta/thetaScale];
end


function labels = loadClusterCsv(clusterFile,busIds)
assert(isfile(clusterFile), 'Cluster CSV does not exist: %s',clusterFile);
T = readtable(clusterFile);
assert(all(ismember({'bus_id','cluster'},T.Properties.VariableNames)), ...
    'Cluster CSV must contain bus_id and cluster columns.');
[found,row] = ismember(busIds,double(T.bus_id));
assert(all(found), 'Cluster CSV does not contain every bus ID.');
labels = double(T.cluster(row));
assert(all(labels == round(labels) & labels >= 1), ...
    'Cluster labels must be positive integers.');
end


function adjacency = caseAdjacency(caseName,busIds)
define_constants;
mpc = loadcase(caseName);
active = mpc.branch(:,BR_STATUS) > 0;
fromBus = mpc.branch(active,F_BUS);
toBus = mpc.branch(active,T_BUS);
[foundF,fromIndex] = ismember(fromBus,busIds);
[foundT,toIndex] = ismember(toBus,busIds);
assert(all(foundF & foundT), ...
    'The MATPOWER case and affine-result bus IDs do not match.');
nb = numel(busIds);
adjacency = sparse([fromIndex;toIndex],[toIndex;fromIndex],1,nb,nb);
adjacency = spones(adjacency);
end


function report = connectivityReport(adjacency,labels,busIds)
K = max(labels);
report = repmat(struct('cluster',0,'number_of_buses',0, ...
    'number_of_components',0,'is_connected',false, ...
    'components',{{}},'component_sizes',[],'component_bus_ids',{{}}),K,1);
for c = 1:K
    nodes = find(labels == c);
    assert(~isempty(nodes), 'Cluster %d is empty.',c);
    components = dfsComponents(adjacency,nodes);
    sizes = cellfun(@numel,components);
    minimumBus = cellfun(@(x) min(busIds(x)),components);
    [~,order] = sortrows([-sizes(:),minimumBus(:)],[1,2]);
    components = components(order);
    sizes = sizes(order);

    report(c).cluster = c;
    report(c).number_of_buses = numel(nodes);
    report(c).number_of_components = numel(components);
    report(c).is_connected = numel(components) == 1;
    report(c).components = components;
    report(c).component_sizes = sizes;
    report(c).component_bus_ids = cellfun( ...
        @(x) busIds(x)',components,'UniformOutput',false);
end
end


function components = dfsComponents(adjacency,allowedNodes)
nb = size(adjacency,1);
unvisited = false(nb,1);
unvisited(allowedNodes) = true;
components = {};
while any(unvisited)
    root = find(unvisited,1);
    stack = root;
    unvisited(root) = false;
    component = zeros(1,0);
    while ~isempty(stack)
        current = stack(end);
        stack(end) = [];
        component(end+1) = current; %#ok<AGROW>
        neighbors = find(adjacency(current,:) > 0);
        neighbors = neighbors(unvisited(neighbors));
        unvisited(neighbors) = false;
        stack = [stack,neighbors]; %#ok<AGROW>
    end
    components{end+1} = component; %#ok<AGROW>
end
end


function best = selectRepairAction(adjacency,labels,X,report,busIds)
K = max(labels);
centers = zeros(K,size(X,2));
for c = 1:K
    centers(c,:) = mean(X(labels == c,:),1);
end
best = blankAction();

for c = 1:K
    if report(c).is_connected, continue; end
    main = report(c).components{1};
    for j = 2:report(c).number_of_components
        island = report(c).components{j};

        % Option 1: move the entire island to a physically adjacent cluster.
        boundary = find(any(adjacency(island,:),1))';
        targets = setdiff(unique(labels(boundary)),c);
        for target = targets(:)'
            action = buildAction('move component',island,target, ...
                labels,X,centers,adjacency);
            best = betterAction(best,action,busIds);
        end

        % Option 2: absorb a minimum-cardinality bridge into this cluster.
        path = minimumForeignPath(adjacency,labels,c,main,island);
        bridge = path(labels(path) ~= c);
        if ~isempty(bridge) && donorClustersStayConnected( ...
                adjacency,labels,bridge)
            action = buildAction('absorb bridge',bridge,c, ...
                labels,X,centers,adjacency);
            best = betterAction(best,action,busIds);
        end
    end
end
end


function path = minimumForeignPath(adjacency,labels,targetCluster,sources,targets)
% Dijkstra with lexicographic cost: foreign buses first, then path length.
nb = size(adjacency,1);
foreignCost = inf(nb,1);
hopCost = inf(nb,1);
previous = zeros(nb,1);
visited = false(nb,1);
foreignCost(sources) = 0;
hopCost(sources) = 0;
targetMask = false(nb,1);
targetMask(targets) = true;
finish = 0;

while true
    candidates = find(~visited & isfinite(foreignCost));
    if isempty(candidates), break; end
    [~,order] = sortrows([foreignCost(candidates),hopCost(candidates)],[1,2]);
    current = candidates(order(1));
    visited(current) = true;
    if targetMask(current)
        finish = current;
        break;
    end
    neighbors = find(adjacency(current,:) > 0);
    for next = neighbors
        proposedForeign = foreignCost(current)+(labels(next) ~= targetCluster);
        proposedHops = hopCost(current)+1;
        improves = proposedForeign < foreignCost(next) || ...
            (proposedForeign == foreignCost(next) && ...
             proposedHops < hopCost(next));
        if improves
            foreignCost(next) = proposedForeign;
            hopCost(next) = proposedHops;
            previous(next) = current;
        end
    end
end
assert(finish > 0, 'No path exists between disconnected components.');

path = finish;
while previous(path(1)) ~= 0
    path = [previous(path(1)),path]; %#ok<AGROW>
end
end


function safe = donorClustersStayConnected(adjacency,labels,movedNodes)
safe = true;
donors = unique(labels(movedNodes));
for donor = donors(:)'
    original = find(labels == donor);
    remaining = setdiff(original,movedNodes);
    if isempty(remaining)
        safe = false;
        return;
    end
    beforeCount = numel(dfsComponents(adjacency,original));
    afterCount = numel(dfsComponents(adjacency,remaining));
    if afterCount > beforeCount
        safe = false;
        return;
    end
end
end


function [labels,moves,rounds] = balanceConnectedClusters( ...
        adjacency,labels,X,busIds,moves,maxRounds,featureWeight, ...
        cutWeight,timeLimit,mipGap)
assert(exist('binvar','file') == 2 && exist('optimize','file') == 2, ...
    'Balanced refinement requires YALMIP.');
K = max(labels);
nb = numel(labels);
lowerSize = floor(nb/K);
upperSize = ceil(nb/K);
initialLabels = labels;

centers = zeros(K,size(X,2));
for c = 1:K
    centers(c,:) = mean(X(labels == c,:),1);
end
featureDistance = zeros(nb,K);
for c = 1:K
    featureDistance(:,c) = sum((X-centers(c,:)).^2,2);
end
positiveDistance = featureDistance(featureDistance > 0);
if isempty(positiveDistance)
    featureScale = 1;
else
    featureScale = max(median(positiveDistance),eps);
end

[fromBus,toBus] = find(triu(adjacency,1));
numberOfEdges = numel(fromBus);
x = binvar(nb,K,'full');
cutEdge = sdpvar(numberOfEdges,1);
C = [sum(x,2) == 1, sum(x,1) >= lowerSize, ...
    sum(x,1) <= upperSize, cutEdge >= 0, cutEdge <= 1];
for c = 1:K
    C = [C, cutEdge >= x(fromBus,c)-x(toBus,c), ...
        cutEdge >= x(toBus,c)-x(fromBus,c)]; %#ok<AGROW>
end

featureObjective = sum(sum(featureDistance.*x))/(nb*featureScale);
cutObjective = sum(cutEdge)/numberOfEdges;
objective = featureWeight*featureObjective+cutWeight*cutObjective;
settings = sdpsettings('solver','gurobi','verbose',1, ...
    'usex0',1,'gurobi.TimeLimit',timeLimit, ...
    'gurobi.MIPGap',mipGap,'gurobi.NumericFocus',1);

rounds = 0;
solved = false;
for iteration = 1:maxRounds
    diagnostics = optimize(C,objective,settings);
    assert(diagnostics.problem == 0, ...
        'Balanced partition MIP failed in round %d: %s', ...
        iteration,diagnostics.info);
    xValue = value(x);
    [~,candidateLabels] = max(xValue,[],2);
    candidateReport = connectivityReport( ...
        adjacency,candidateLabels,busIds);
    fprintf(['Balance MIP round %d: objective %.6g, ', ...
        'cut edges %d, disconnected clusters %d\n'], ...
        iteration,value(objective),round(value(sum(cutEdge))), ...
        nnz(~[candidateReport.is_connected]));
    rounds = iteration;
    if all([candidateReport.is_connected])
        labels = candidateLabels;
        solved = true;
        break;
    end

    % Exclude every disconnected component unless it connects through at
    % least one physical boundary bus assigned to the same cluster.
    for c = 1:K
        if candidateReport(c).is_connected, continue; end
        for component = candidateReport(c).components
            nodes = component{1};
            boundary = find(any(adjacency(nodes,:),1))';
            boundary = setdiff(boundary,nodes);
            assert(~isempty(boundary), ...
                'Cluster %d contains a component with no physical boundary.',c);
            for node = nodes(:)'
                C = [C, sum(x(boundary,c)) >= x(node,c)]; %#ok<AGROW>
            end
        end
    end
    assign(x,round(xValue));
    assign(cutEdge,value(cutEdge));
end

assert(solved, ...
    'Balanced MIP did not produce connected clusters in %d rounds.',maxRounds);

for node = find(labels ~= initialLabels)'
    action = blankAction();
    action.valid = true;
    action.type = 'balanced topology MIP';
    action.nodes = node;
    action.to_cluster = labels(node);
    action.number_changed = 1;
    oldCenter = centers(initialLabels(node),:);
    newCenter = centers(labels(node),:);
    action.feature_delta = sum((X(node,:)-newCenter).^2)- ...
        sum((X(node,:)-oldCenter).^2);
    action.boundary_edges = nnz(adjacency(node,labels == labels(node)));
    step = numel(moves)+1;
    moves(end+1) = makeMoveRecord( ...
        step,action,initialLabels(node),busIds); %#ok<AGROW>
end
end


function action = buildAction(type,nodes,target,labels,X,centers,adjacency)
nodes = unique(nodes(:));
oldLabels = labels(nodes);
oldDistance = 0;
for j = 1:numel(nodes)
    oldDistance = oldDistance+sum((X(nodes(j),:)-centers(oldLabels(j),:)).^2);
end
newDistance = sum(sum((X(nodes,:)-centers(target,:)).^2,2));
action = blankAction();
action.valid = true;
action.type = type;
action.nodes = nodes;
action.to_cluster = target;
action.number_changed = numel(nodes);
action.feature_delta = newDistance-oldDistance;
action.boundary_edges = nnz(adjacency(nodes,labels == target));
end


function best = betterAction(best,candidate,busIds)
if ~candidate.valid, return; end
if ~best.valid
    best = candidate;
    return;
end
candidateKey = [candidate.number_changed,candidate.feature_delta, ...
    -candidate.boundary_edges,min(busIds(candidate.nodes)),candidate.to_cluster];
bestKey = [best.number_changed,best.feature_delta, ...
    -best.boundary_edges,min(busIds(best.nodes)),best.to_cluster];
firstDifference = find(abs(candidateKey-bestKey) > 1e-12,1);
if ~isempty(firstDifference) && ...
        candidateKey(firstDifference) < bestKey(firstDifference)
    best = candidate;
end
end


function action = blankAction()
action = struct('valid',false,'type','','nodes',zeros(0,1), ...
    'to_cluster',0,'number_changed',Inf,'feature_delta',Inf, ...
    'boundary_edges',0,'cut_delta',Inf,'score',Inf);
end


function moves = emptyMoveLog()
moves = repmat(struct('step',0,'action','','bus_ids','', ...
    'from_clusters','','to_cluster',0,'number_changed',0, ...
    'feature_delta',0,'boundary_edges',0),0,1);
end


function record = makeMoveRecord(step,action,oldLabels,busIds)
record.step = step;
record.action = action.type;
record.bus_ids = char(strjoin(string(busIds(action.nodes)'),', '));
record.from_clusters = char(strjoin(string(unique(oldLabels)'),', '));
record.to_cluster = action.to_cluster;
record.number_changed = action.number_changed;
record.feature_delta = action.feature_delta;
record.boundary_edges = action.boundary_edges;
end


function printConnectivity(report,titleText)
fprintf('\n%s\n',titleText);
for c = 1:numel(report)
    fprintf('Cluster %d: %d buses, %d component(s)', ...
        c,report(c).number_of_buses,report(c).number_of_components);
    if report(c).is_connected
        fprintf(' -- connected\n');
    else
        fprintf(' -- DISCONNECTED\n');
        for j = 1:report(c).number_of_components
            buses = report(c).component_bus_ids{j};
            fprintf('  Component %d (%d buses): %s\n',j,numel(buses), ...
                strjoin(string(buses),', '));
        end
    end
end
end


function writeConnectivity(report,fileName)
K = numel(report);
cluster = (1:K)';
numberOfBuses = [report.number_of_buses]';
numberOfComponents = [report.number_of_components]';
isConnected = [report.is_connected]';
components = strings(K,1);
for c = 1:K
    text = strings(report(c).number_of_components,1);
    for j = 1:report(c).number_of_components
        text(j) = "["+strjoin(string(report(c).component_bus_ids{j})," ")+"]";
    end
    components(c) = strjoin(text," | ");
end
T = table(cluster,numberOfBuses,numberOfComponents,isConnected,components, ...
    'VariableNames',{'cluster','number_of_buses','number_of_components', ...
    'is_connected','components'});
writetable(T,fileName);
end


function writeMoves(moves,fileName)
if isempty(moves)
    T = table('Size',[0,8], ...
        'VariableTypes',{'double','string','string','string','double', ...
        'double','double','double'}, ...
        'VariableNames',{'step','action','bus_ids','from_clusters', ...
        'to_cluster','number_changed','feature_delta','boundary_edges'});
else
    T = struct2table(moves);
end
writetable(T,fileName);
end


function writeClusterJson(labels,busIds,fileName)
parent = fileparts(fileName);
if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
K = max(labels);
clusters = cell(K,1);
for c = 1:K
    clusters{c} = reshape(busIds(labels == c),1,[]);
end
payload = struct('clusters',{clusters});
try
    jsonText = jsonencode(payload,'PrettyPrint',true);
catch
    jsonText = jsonencode(payload);
end
fileId = fopen(fileName,'w');
assert(fileId >= 0, 'Cannot open JSON output: %s',fileName);
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
fprintf(fileId,'%s\n',jsonText);
end
