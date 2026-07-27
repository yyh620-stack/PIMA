function r = solve_bus_affine_g(o)
%SOLVE_PIMA_BUS_AFFINE_G PIMA bus-state and generator redispatch matrices.
%
% Each bus has one independent signed uncertainty in [-radius,+radius].
% Its P/Q net injections change together. Output columns are
% [lower bus 1..N, upper bus 1..N]. Requires MATPOWER, YALMIP, and Gurobi.
%
%   r = solve_pima_bus_affine_g;
%   r = solve_pima_bus_affine_g(radius=0.15, w_pg=1, w_vm=100);

arguments
    o.case_name (1,1) string = "case118"
    o.output_dir (1,1) string = "pima_bus_affine_results"
    o.radius (1,1) double {mustBeNonnegative,mustBeFinite} = 0.15
    o.time_limit (1,1) double {mustBePositive,mustBeFinite} = 600
    % Objective-weight entry: active redispatch is primary; the rest
    % regularize reactive power, voltage, angle, and line-flow responses.
    o.w_pg (1,1) double {mustBeNonnegative,mustBeFinite} = 1.0
    o.w_qg (1,1) double {mustBeNonnegative,mustBeFinite} = 0.20
    o.w_vm (1,1) double {mustBeNonnegative,mustBeFinite} = 100
    o.w_va (1,1) double {mustBeNonnegative,mustBeFinite} = 10
    o.w_flow (1,1) double {mustBeNonnegative,mustBeFinite} = 0.05
end

assert(exist('runopf','file') == 2, 'MATPOWER is required.');
assert(exist('sdpvar','file') == 2, 'YALMIP is required.');
define_constants;

%% AC-OPF base point and first-order AC model
mpopt = mpoption('verbose',0,'out.all',0);
base = runopf(char(o.case_name),mpopt);
assert(base.success == 1, 'The base AC-OPF did not converge.');
base = ext2int(base);
bus = base.bus; gen = base.gen; branch = base.branch;
baseMVA = base.baseMVA;
nb = size(bus,1); ng = size(gen,1); nl = size(branch,1);

[Ybus,Yf,Yt] = makeYbus(baseMVA,bus,branch);
V0 = bus(:,VM).*exp(1j*deg2rad(bus(:,VA)));
fbus = branch(:,F_BUS); tbus = branch(:,T_BUS);
[J,Sf0,St0,Hf,Ht] = linearizeAC(Ybus,Yf,Yt,fbus,tbus,V0,1e-6);

Cg = sparse(gen(:,GEN_BUS),1:ng,1,nb,ng);
Cgen = blkdiag(Cg,Cg);
pg0 = gen(:,PG)/baseMVA; qg0 = gen(:,QG)/baseMVA;
pNet0 = Cg*pg0 - bus(:,PD)/baseMVA;
qNet0 = Cg*qg0 - bus(:,QD)/baseMVA;
E = o.radius*[diag(pNet0);diag(qNet0)];

%% Affine state and corrective-generation policies
% State order: [theta_1..theta_N, Vm_1..Vm_N].
% Generator order: [Pg_1..Pg_G, Qg_1..Qg_G].
AstatePos = sdpvar(2*nb,nb,'full'); AstateNeg = sdpvar(2*nb,nb,'full');
AgenPos = sdpvar(2*ng,nb,'full'); AgenNeg = sdpvar(2*ng,nb,'full');
Astate = AstatePos-AstateNeg; AstateAbs = AstatePos+AstateNeg;
Agen = AgenPos-AgenNeg; AgenAbs = AgenPos+AgenNeg;
Atheta = Astate(1:nb,:); Avm = Astate(nb+1:end,:);
Apg = Agen(1:ng,:); Aqg = Agen(ng+1:end,:);

C = [J*Astate == Cgen*Agen + E, AstatePos >= 0, AstateNeg >= 0, AgenPos >= 0, AgenNeg >= 0];
ref = find(bus(:,BUS_TYPE) == REF);
C = [C, Atheta(ref,:) == 0];

onlineGen = gen(:,GEN_STATUS) > 0;
pmin = gen(:,PMIN)/baseMVA; pmax = gen(:,PMAX)/baseMVA;
qmin = gen(:,QMIN)/baseMVA; qmax = gen(:,QMAX)/baseMVA;
pmin(~onlineGen)=0; pmax(~onlineGen)=0;
qmin(~onlineGen)=0; qmax(~onlineGen)=0;
rPg = sum(AgenAbs(1:ng,:),2); rQg = sum(AgenAbs(ng+1:end,:),2);
rVm = sum(AstateAbs(nb+1:end,:),2);
C = [C, pg0-rPg >= pmin, pg0+rPg <= pmax, qg0-rQg >= qmin, qg0+rQg <= qmax, bus(:,VM)-rVm >= bus(:,VMIN), bus(:,VM)+rVm <= bus(:,VMAX)];

incidence = sparse((1:nl)',fbus,1,nl,nb) + sparse((1:nl)',tbus,-1,nl,nb);
angle0 = incidence*deg2rad(bus(:,VA));
boundedAngle = branch(:,BR_STATUS)>0 & branch(:,ANGMIN)>-360 & branch(:,ANGMAX)<360;
if any(boundedAngle)
    angleResponse = incidence(boundedAngle,:)*Atheta;
    angleAbs = sdpvar(nnz(boundedAngle),nb,'full');
    C = [C, angleAbs >= angleResponse, angleAbs >= -angleResponse, angleAbs >= 0];
    rAngle = sum(angleAbs,2);
    C = [C, angle0(boundedAngle)-rAngle >= ...
        deg2rad(branch(boundedAngle,ANGMIN)), ...
        angle0(boundedAngle)+rAngle <= deg2rad(branch(boundedAngle,ANGMAX))];
end

% First-order apparent-flow magnitude at both ends of each rated branch.
unitF = conj(Sf0)./max(abs(Sf0),1e-8);
unitT = conj(St0)./max(abs(St0),1e-8);
HmagF = real(unitF.*Hf); HmagT = real(unitT.*Ht);
rated = branch(:,BR_STATUS)>0 & branch(:,RATE_A)>0;
if any(rated)
    Hflow = [HmagF(rated,:);HmagT(rated,:)];
    flow0 = [abs(Sf0(rated));abs(St0(rated))];
    flowLimit = repmat(branch(rated,RATE_A)/baseMVA,2,1);
    flowResponse = Hflow*Astate;
    flowAbs = sdpvar(2*nnz(rated),nb,'full');
    C = [C, flowAbs >= flowResponse, flowAbs >= -flowResponse, flowAbs >= 0, flow0 + sum(flowAbs,2) <= flowLimit];
    flowPenalty = baseMVA*sum(flowAbs(:))/numel(flowAbs);
else
    Hflow = zeros(0,2*nb); flow0 = zeros(0,1); flowLimit = zeros(0,1);
    flowPenalty = 0;
    fprintf('No positive RATE_A values: line limits and w_flow are inactive.\n');
end

%% Paper-compatible operation objective plus stable-policy regularization
% The paper leaves f(z,G) open (operation or expected cost). The AC-OPF
% already minimizes base cost, so these normalized L1 terms select a stable
% feasible policy, prioritizing active-power corrective redispatch.
objective = o.w_pg*baseMVA*sum(sum(AgenAbs(1:ng,:)))/numel(Apg) + ...
    o.w_qg*baseMVA*sum(sum(AgenAbs(ng+1:end,:)))/numel(Aqg) + ...
    o.w_vm*sum(sum(AstateAbs(nb+1:end,:)))/numel(Avm) + ...
    o.w_va*sum(sum(AstateAbs(1:nb,:)))/numel(Atheta) + ...
    o.w_flow*flowPenalty;

settings = sdpsettings('solver','gurobi','verbose',1, ...
    'gurobi.TimeLimit',o.time_limit,'gurobi.Method',2, ...
    'gurobi.BarHomogeneous',1,'gurobi.Crossover',0, ...
    'gurobi.NumericFocus',1,'gurobi.ScaleFlag',2);
fprintf('Compiling and solving the affine LP (%d buses, %d generators)...\n',nb,ng);
solveTimer = tic;
diagnostics = optimize(C,objective,settings);
fprintf('YALMIP + Gurobi elapsed time: %.1f s\n',toc(solveTimer));
assert(diagnostics.problem == 0, 'Gurobi failed: %s',diagnostics.info);

%% Published bus layout and generator redispatch layout
As = value(Astate); Ag = value(Agen);
GstateGrouped = [-As,As];
GgenGrouped = baseMVA*[-Ag,Ag];
stateRows = reshape([nb+(1:nb);1:nb],[],1);   % [Vm_i; theta_i]
genRows = reshape([1:ng;ng+(1:ng)],[],1);    % [Pg_g; Qg_g]

r.G_state = GstateGrouped(stateRows,:);
r.G_redispatch = GgenGrouped(genRows,:);
r.G_state_grouped = GstateGrouped;
r.G_redispatch_grouped = GgenGrouped;
r.base_voltage = [bus(:,VM),bus(:,VA)];
r.base_generation = [gen(:,PG),gen(:,QG)];
r.base_net_injection = baseMVA*[pNet0,qNet0];
r.bus_ids = bus(:,BUS_I); r.generator_buses = gen(:,GEN_BUS);
r.radius = o.radius;
r.column_order = '[LB_bus_1..N, UB_bus_1..N]';
r.state_row_order = '[Vm_1,theta_1,Vm_2,theta_2,...] (pu, rad)';
r.redispatch_row_order = '[Pg_1,Qg_1,Pg_2,Qg_2,...] (MW, MVAr)';
r.objective = value(objective);
r.weights = struct('pg',o.w_pg,'qg',o.w_qg,'vm',o.w_vm, 'va',o.w_va,'flow',o.w_flow);

balanceResidual = J*As-Cgen*Ag-E;
r.validation.linear_balance_residual = max(abs(balanceResidual),[],'all');
r.validation.generator_violation_pu = max([0; ...
    pg0+sum(abs(Ag(1:ng,:)),2)-pmax; ...
    pmin-pg0+sum(abs(Ag(1:ng,:)),2); ...
    qg0+sum(abs(Ag(ng+1:end,:)),2)-qmax; ...
    qmin-qg0+sum(abs(Ag(ng+1:end,:)),2)]);
r.validation.voltage_violation_pu = max([0; ...
    bus(:,VM)+sum(abs(As(nb+1:end,:)),2)-bus(:,VMAX); ...
    bus(:,VMIN)-bus(:,VM)+sum(abs(As(nb+1:end,:)),2)]);
r.validation.flow_violation_mva = baseMVA*max([0; ...
    flow0+sum(abs(Hflow*As),2)-flowLimit]);

if ~isfolder(o.output_dir), mkdir(o.output_dir); end
writematrix(r.G_state,fullfile(o.output_dir,'G_state.csv'));
writematrix(r.G_redispatch,fullfile(o.output_dir,'G_redispatch.csv'));
save(fullfile(o.output_dir,'pima_affine_solution.mat'),'r');

fprintf('PIMA uncertainty: each bus net injection +/- %.1f%%\n',100*o.radius);
fprintf('G_state: %d x %d; G_redispatch: %d x %d\n', size(r.G_state),size(r.G_redispatch));
fprintf('Max linear balance residual: %.3e\n', r.validation.linear_balance_residual);
end


function [J,Sf0,St0,Hf,Ht] = linearizeAC(Ybus,Yf,Yt,fbus,tbus,V,h)
nb = numel(V); nl = numel(fbus);
S0 = V.*conj(Ybus*V);
Sf0 = V(fbus).*conj(Yf*V);
St0 = V(tbus).*conj(Yt*V);
J = zeros(2*nb); Hf = complex(zeros(nl,2*nb)); Ht = Hf;
for k = 1:2*nb
    Vp = V;
    if k <= nb
        Vp(k) = V(k)*exp(1j*h);
    else
        i = k-nb;
        Vp(i) = (abs(V(i))+h)*exp(1j*angle(V(i)));
    end
    Sp = Vp.*conj(Ybus*Vp);
    Sfp = Vp(fbus).*conj(Yf*Vp);
    Stp = Vp(tbus).*conj(Yt*Vp);
    J(:,k) = [real(Sp-S0);imag(Sp-S0)]/h;
    Hf(:,k) = (Sfp-Sf0)/h;
    Ht(:,k) = (Stp-St0)/h;
end
% Finite differences leave tiny numerical fill-in; removing it preserves
% the physical network sparsity and greatly reduces YALMIP compile time.
J(abs(J)<1e-9)=0; Hf(abs(Hf)<1e-9)=0; Ht(abs(Ht)<1e-9)=0;
J = sparse(J); Hf = sparse(Hf); Ht = sparse(Ht);
end
