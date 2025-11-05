function X = OC(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp)
l1 = 0; l2 = 1e9; move = 0.2;
while (l2-l1)/(l1+l2) > 1e-3
    lmid = 0.5*(l2+l1);
    X.CtrPts_new = max(0,max(X.CtrPts-move,min(1,min(X.CtrPts+move,X.CtrPts.*sqrt(-dJ_dp./dv_dp/lmid)))));
    X.CtrPts_new = (Sh*X.CtrPts_new)./Hs;
    X.GauPts = R*X.CtrPts_new;
    if mean(X.GauPts(:)) > Vmax
        l1 = lmid;
    else
        l2 = lmid;
    end
end
end