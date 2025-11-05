function [KE, dKE, dv_dg] = Stiff_Ele2D(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv)
KE = cell(Ele.Num,1);
dKE = cell(Ele.Num,1);
dv_dg = zeros(GauPts.Num,1);
Nen = Ele.CtrPtsNum;

% Debug info: print basic parameters
fprintf('=== Stiff_Ele2D Debug Info ===\n');
fprintf('Ele.Num = %d, Ele.CtrPtsNum = %d, GauPts.Num = %d\n', Ele.Num, Ele.CtrPtsNum, GauPts.Num);
fprintf('dRu size: [%d, %d], dRv size: [%d, %d]\n', size(dRu,1), size(dRu,2), size(dRv,1), size(dRv,2));
fprintf('CtrPts.Cordis size: [%d, %d]\n', size(CtrPts.Cordis,1), size(CtrPts.Cordis,2));

neg_det_count_J1 = 0;
neg_det_count_J = 0;

for ide = 1:Ele.Num
    [idv, idu] = find(Ele.Seque == ide);                    % The two idices in two parametric directions for an element
    Ele_Knot_U = Ele.KnotsU(idu,:);                         % The knot span in the first parametric direction for an element
    Ele_Knot_V = Ele.KnotsV(idv,:);                         % The knot span in the second parametric direction for an element
    Ele_NoCtPt = Ele.CtrPtsCon(ide,:);                      % The number of control points in an element
    Ele_CoCtPt = CtrPts.Cordis(1:2,Ele_NoCtPt);             % The coordinates of the control points in an element
    
    % Debug info: print element information
    if ide <= 3 || mod(ide, 500) == 0
        fprintf('\n--- Element %d ---\n', ide);
        fprintf('  idu=%d, idv=%d\n', idu, idv);
        fprintf('  Ele_Knot_U = [%.6f, %.6f]\n', Ele_Knot_U(1), Ele_Knot_U(2));
        fprintf('  Ele_Knot_V = [%.6f, %.6f]\n', Ele_Knot_V(1), Ele_Knot_V(2));
        fprintf('  Ele_NoCtPt (first 5): %s\n', mat2str(Ele_NoCtPt(1:min(5,length(Ele_NoCtPt)))));
        fprintf('  Ele_CoCtPt size: [%d, %d]\n', size(Ele_CoCtPt,1), size(Ele_CoCtPt,2));
    end
    
    Ke = zeros(2*Nen,2*Nen);
    dKe = cell(Ele.GauPtsNum,1);
    for i = 1:Ele.GauPtsNum
        GptOrder = GauPts.Seque(ide, i);
        
        % Debug info: print dRu and dRv information (only for first few elements and first Gauss point)
        if (ide <= 2 && i == 1) || (ide == 1 && i <= 2)
            fprintf('  [Element %d, Gauss point %d] GptOrder=%d\n', ide, i, GptOrder);
            fprintf('    dRu(GptOrder, 1:%d) = %s\n', min(5,size(dRu,2)), mat2str(dRu(GptOrder, 1:min(5,size(dRu,2)))));
            fprintf('    dRv(GptOrder, 1:%d) = %s\n', min(5,size(dRv,2)), mat2str(dRv(GptOrder, 1:min(5,size(dRv,2)))));
            fprintf('    Ele_NoCtPt = %s\n', mat2str(Ele_NoCtPt));
        end
        
        dR_dPara = [dRu(GptOrder,:); dRv(GptOrder,:)];
        dPhy_dPara = dR_dPara*Ele_CoCtPt';
        J1 = dPhy_dPara;
        
        % Debug info: check determinant of J1
        det_J1 = det(J1);
        if det_J1 < 0
            neg_det_count_J1 = neg_det_count_J1 + 1;
            if neg_det_count_J1 <= 5 || (ide <= 3 && i <= 2)
                fprintf('  [Element %d, Gauss point %d] J1 determinant is negative: det=%.6e\n', ide, i, det_J1);
                fprintf('    J1 = [%.6e, %.6e; %.6e, %.6e]\n', J1(1,1), J1(1,2), J1(2,1), J1(2,2));
                fprintf('    GptOrder=%d, dR_dPara size: [%d, %d]\n', GptOrder, size(dR_dPara,1), size(dR_dPara,2));
                fprintf('    dR_dPara(1, 1:3) = %s\n', mat2str(dR_dPara(1, 1:min(3,size(dR_dPara,2)))));
                fprintf('    dR_dPara(2, 1:3) = %s\n', mat2str(dR_dPara(2, 1:min(3,size(dR_dPara,2)))));
                fprintf('    Ele_CoCtPt (first 2 cols): [%.6e, %.6e; %.6e, %.6e]\n', ...
                    Ele_CoCtPt(1,1), Ele_CoCtPt(1,2), Ele_CoCtPt(2,1), Ele_CoCtPt(2,2));
            end
        end
        
        dR_dPhy = inv(J1)*dR_dPara;
        Be(1,1:Nen) = dR_dPhy(1,:); Be(2,Nen+1:2*Nen) = dR_dPhy(2,:);
        Be(3,1:Nen) = dR_dPhy(2,:); Be(3,Nen+1:2*Nen) = dR_dPhy(1,:);
        dPara_dPare(1,1) = (Ele_Knot_U(2)-Ele_Knot_U(1))/2; % the mapping from the parametric space to the parent space
        dPara_dPare(2,2) = (Ele_Knot_V(2)-Ele_Knot_V(1))/2;
        J2 = dPara_dPare;  J = J1*J2;                       % the mapping from the physical space to the parent space;
        
        % Debug info: check determinant of J
        det_J = det(J);
        if det_J < 0
            neg_det_count_J = neg_det_count_J + 1;
            if neg_det_count_J <= 5 || (ide <= 3 && i <= 2)
                fprintf('  [Element %d, Gauss point %d] J determinant is negative: det=%.6e\n', ide, i, det_J);
                fprintf('    J2 = [%.6e, 0; 0, %.6e]\n', J2(1,1), J2(2,2));
            end
        end
        
        weight = GauPts.Weigh(i)*det(J);                    % Weight factor at this point
        Ke = Ke + (Emin+X.GauPts(GptOrder,:).^penal*(1-Emin))*weight*(Be'*DH*Be);
        dKe{i} = (penal*X.GauPts(GptOrder,:).^(penal-1)*(1-Emin))*weight*(Be'*DH*Be);
        dv_dg(GptOrder) = weight;
    end
    KE{ide} = Ke;
    dKE{ide} = dKe;
end

% Print statistics
fprintf('\n=== Stiff_Ele2D Statistics ===\n');
fprintf('J1 negative determinant count: %d / %d (%.2f%%)\n', neg_det_count_J1, Ele.Num*Ele.GauPtsNum, 100*neg_det_count_J1/(Ele.Num*Ele.GauPtsNum));
fprintf('J negative determinant count: %d / %d (%.2f%%)\n', neg_det_count_J, Ele.Num*Ele.GauPtsNum, 100*neg_det_count_J/(Ele.Num*Ele.GauPtsNum));
fprintf('==============================\n\n');
end
%======================================================================================================================%
% Subfunction Stiff_Ele2D:                                                                                             %
%                                                                                                                      %
% A compact and efficient MATLAB code for the numerical implementation isogeometric topology optimization in 2D        %
%                                                                                                                      %
% Developed by: Jie Gao                                                                                                %
% Email: JieGao@hust.edu.cn                                                                                            %
%                                                                                                                      %
% Main references:                                                                                                     %
%                                                                                                                      %
% (1) Jie Gao, Lin Wang, Zhen Luo, Liang Gao. IgaTop: an implementation of topology optimization for structures        %
% using IGA in Matlab. Structural and multidisciplinary optimization.                                                  %
%                                                                                                                      %
% (2) Jie Gao, Liang Gao, Zhen Luo, Peigen Li. Isogeometric topology optimization for continuum structures using       %
% density distribution function. Int J Numer Methods Eng, 2019, 119:991�C1017                                          %
%                                                                                                                      %
% *********************************************   Disclaimer   ******************************************************* %
% The authors reserve all rights for the programs. The programs may be distributed and used for academic and           %
% educational purposes. The authors do not guarantee that the code is free from errors,and they shall not be liable    %
% in any event caused by the use of the program.                                                                       %
%======================================================================================================================%