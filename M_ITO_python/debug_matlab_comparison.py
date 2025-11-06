"""
调试脚本 - 生成MATLAB对比数据
用于在MATLAB中运行，生成中间数据供Python对比
"""

matlab_debug_script = """
% MATLAB调试脚本 - 保存中间数据
function Debug_IgaTop2D()
    % 使用小规模参数进行测试
    L = 10; W = 5; Order = [1 1]; Num = [11 6]; BoundCon = 1;
    
    % 材料属性
    E0 = 1; Emin = 1e-9; nu = 0.3; 
    DH = E0/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];
    
    % 生成几何模型
    NURBS = Geom_Mod(L, W, Order, Num, BoundCon);
    
    % IGA预处理
    [CtrPts, Ele, GauPts] = Pre_IGA(NURBS);
    Dim = numel(NURBS.order);
    Dofs.Num = Dim*CtrPts.Num;
    
    % 保存NURBS数据
    save('debug_nurbs.mat', 'NURBS', 'CtrPts', 'Ele', 'GauPts', 'Dim', 'Dofs', 'DH', ...
         'E0', 'Emin', 'nu', 'L', 'W', 'Order', 'Num', 'BoundCon');
    
    fprintf('NURBS数据已保存到 debug_nurbs.mat\\n');
    fprintf('  CtrPts.Num = %d\\n', CtrPts.Num);
    fprintf('  Ele.Num = %d\\n', Ele.Num);
    fprintf('  GauPts.Num = %d\\n', GauPts.Num);
    fprintf('  Ele.Seque(1:3,1:3):\\n');
    disp(Ele.Seque(1:3,1:3));
    fprintf('  Ele.CtrPtsCon(1:3,:):\\n');
    disp(Ele.CtrPtsCon(1:3,:));
    
    % 准备高斯点坐标
    GauPts.Cor = [reshape(GauPts.CorU',1,GauPts.Num); reshape(GauPts.CorV',1,GauPts.Num)];
    
    % 计算基函数
    [N, id] = nrbbasisfun(GauPts.Cor, NURBS);
    fprintf('\\n基函数计算:\\n');
    fprintf('  N size: [%d, %d]\\n', size(N,1), size(N,2));
    fprintf('  id size: [%d, %d]\\n', size(id,1), size(id,2));
    fprintf('  N(1,:) = \\n'); disp(N(1,:));
    fprintf('  id(1,:) = '); disp(id(1,:));
    
    % 计算基函数导数
    [dRu, dRv] = nrbbasisfunder(GauPts.Cor, NURBS);
    fprintf('\\n基函数导数计算:\\n');
    fprintf('  dRu size: [%d, %d]\\n', size(dRu,1), size(dRu,2));
    fprintf('  dRv size: [%d, %d]\\n', size(dRv,1), size(dRv,2));
    fprintf('  dRu(1,:) = \\n'); disp(dRu(1,:));
    fprintf('  dRv(1,:) = \\n'); disp(dRv(1,:));
    
    % 保存基函数数据
    save('debug_basisfun.mat', 'N', 'id', 'dRu', 'dRv', 'GauPts');
    fprintf('\\n基函数数据已保存到 debug_basisfun.mat\\n');
    
    % 初始化设计变量
    X.CtrPts = ones(CtrPts.Num,1);
    X.GauPts = zeros(GauPts.Num,1);
    
    % 构建R矩阵
    R = zeros(GauPts.Num,CtrPts.Num);
    for i = 1:GauPts.Num
        R(i,id(i,:)) = N(i,:);
    end
    R = sparse(R);
    X.GauPts = R*X.CtrPts;
    
    % 设置边界条件
    [DBoudary, F] = Boun_Cond(CtrPts, BoundCon, NURBS, Dofs.Num);
    
    % 计算第一次迭代
    penal = 3;
    [KE, dKE, dv_dg] = Stiff_Ele2D(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv);
    
    fprintf('\\n第一次迭代数据:\\n');
    fprintf('  KE{1} size: [%d, %d]\\n', size(KE{1},1), size(KE{1},2));
    fprintf('  KE{1}(1:3,1:3):\\n'); disp(KE{1}(1:3,1:3));
    fprintf('  dv_dg(1:5) = '); disp(dv_dg(1:5)');
    
    % 组装刚度矩阵
    [K] = Stiff_Ass2D(KE, CtrPts, Ele, Dim, Dofs.Num);
    fprintf('  K size: [%d, %d], nnz = %d\\n', size(K,1), size(K,2), nnz(K));
    
    % 求解
    U = Solving(CtrPts, DBoudary, Dofs, K, F, BoundCon);
    fprintf('  U size: [%d, %d]\\n', size(U,1), size(U,2));
    fprintf('  U(1:10) = '); disp(U(1:10)');
    fprintf('  max(abs(U)) = %.6e\\n', max(abs(U)));
    
    % 计算目标函数
    J = 0;
    for ide = 1:Ele.Num
        Ele_NoCtPt = Ele.CtrPtsCon(ide,:);
        edof = [Ele_NoCtPt,Ele_NoCtPt+CtrPts.Num];
        Ue = U(edof,1);
        J = J + Ue'*KE{ide}*Ue;
    end
    fprintf('  目标函数 J = %.6f\\n', J);
    
    % 保存第一次迭代数据
    save('debug_iteration1.mat', 'X', 'R', 'KE', 'dKE', 'dv_dg', 'K', 'F', ...
         'U', 'J', 'DBoudary', 'penal');
    fprintf('\\n第一次迭代数据已保存到 debug_iteration1.mat\\n');
end
"""

print("请在MATLAB中运行以下代码以生成对比数据：")
print("="*80)
print(matlab_debug_script)
print("="*80)
print("\n或者将上述代码保存为 M_ITO/Debug_IgaTop2D.m 并在MATLAB中运行")

