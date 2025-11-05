function IgaTop2D(L, W, Order, Num, BoundCon, Vmax, penal, rmin)
%% Material properties
path = genpath(pwd); addpath(path); 
E0 = 1; Emin = 1e-9; nu = 0.3; DH=E0/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];
NURBS = Geom_Mod(L, W, Order, Num, BoundCon); close all
%% Preparation for IGA
[CtrPts, Ele, GauPts] = Pre_IGA(NURBS);
Dim = numel(NURBS.order); Dofs.Num = Dim*CtrPts.Num;
[DBoudary, F] = Boun_Cond(CtrPts, BoundCon, NURBS, Dofs.Num);
%% Initialization of control design variables
X.CtrPts = ones(CtrPts.Num,1);
GauPts.Cor = [reshape(GauPts.CorU',1,GauPts.Num); reshape(GauPts.CorV',1,GauPts.Num)];
[GauPts.PCor,GauPts.Pw] = nrbeval(NURBS, GauPts.Cor);
GauPts.PCor = GauPts.PCor./GauPts.Pw;
[N, id] = nrbbasisfun(GauPts.Cor, NURBS);
R = zeros(GauPts.Num,CtrPts.Num);
for i = 1:GauPts.Num, R(i,id(i,:)) = N(i,:); end
R = sparse(R);
[dRu, dRv] = nrbbasisfunder(GauPts.Cor, NURBS);

% Debug info: print nrbbasisfunder output information
fprintf('=== nrbbasisfunder Debug Info ===\n');
fprintf('dRu size: [%d, %d]\n', size(dRu,1), size(dRu,2));
fprintf('dRv size: [%d, %d]\n', size(dRv,1), size(dRv,2));
fprintf('GauPts.Num = %d\n', GauPts.Num);
if size(dRu,1) > 0 && size(dRu,2) > 0
    fprintf('dRu(1, 1:5) = %s\n', mat2str(dRu(1, 1:min(5,size(dRu,2)))));
    fprintf('dRv(1, 1:5) = %s\n', mat2str(dRv(1, 1:min(5,size(dRv,2)))));
end
fprintf('=================================\n\n');
X.GauPts = R*X.CtrPts;
%% Smoothing mechanism
[Sh, Hs] = Shep_Fun(CtrPts, rmin);
%% Start optimization in a loop
change = 1; nloop = 150; Data = zeros(nloop,2); Iter_Ch = zeros(nloop,1);
[DenFied, Pos] = Plot_Data(Num, NURBS);
for loop = 1:nloop
    %% IGA to evaluate the displacement responses
    [KE, dKE, dv_dg] = Stiff_Ele2D(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv);
    [K] = Stiff_Ass2D(KE, CtrPts, Ele, Dim, Dofs.Num);
    U = Solving(CtrPts, DBoudary, Dofs, K, F, BoundCon);
    %% Objective function and sensitivity analysis
    J = 0;
    dJ_dg = zeros(GauPts.Num,1);
    for ide = 1:Ele.Num
        Ele_NoCtPt = Ele.CtrPtsCon(ide,:);
        edof = [Ele_NoCtPt,Ele_NoCtPt+CtrPts.Num];
        Ue = U(edof,1);
        J = J + Ue'*KE{ide}*Ue;
        for i = 1:Ele.GauPtsNum
            GptOrder = GauPts.Seque(ide, i);
            dJ_dg(GptOrder) = -Ue'*dKE{ide}{i}*Ue;
        end
    end
    Data(loop,1) = J; Data(loop,2) = mean(X.GauPts(:));
    dJ_dp = R'*dJ_dg; dJ_dp = Sh*(dJ_dp./Hs);
    dv_dp = R'*dv_dg; dv_dp = Sh*(dv_dp./Hs);
    %% Print and plot results
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,J,mean(X.GauPts(:)),change);
    [X] = Plot_Topy(X, GauPts, CtrPts, DenFied, Pos);
    if change < 0.01, break; end
    %% Optimality criteria to update design variables
    X = OC(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp);
    change = max(abs(X.CtrPts_new(:)-X.CtrPts(:))); Iter_Ch(loop) = change;
    X.CtrPts = X.CtrPts_new;
end
end
