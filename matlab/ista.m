function result=ista(x,opts)

result=struct();
lam1=opts.lam1;
alph=opts.alph;
D=opts.D;
ftol=opts.ftol; %stop iterations after cost function achieves this tol.

verbose=1;
if isfield(opts,'verbose')
    verbose=opts.verbose;
end

f_mse=@(x,h)0.5*norm(x-D*h,2)^2;
f_reg=@(h)lam1*norm(h,1);

ftol_cur=Inf;
niter=0;
% initial h
if isfield(opts,'hinit')
    hprev=opts.hinit;
else
%     hprev=randn(size(D,2),1);
    hprev=zeros(size(D,2),1);
end
% DtD=(D')*D;
% Dtx_over_alph=((D')*x)./alph;
Dt=(D');
mse_all(1)=f_mse(x,hprev);
freg_all(1)=f_reg(hprev);
fprev=mse_all(1)+freg_all(1);
f_all(1)=fprev;
done=0;
while ~done
    
    niter=niter+1;
    
%     hcur = soft( hprev - (DtD*hprev)./alph + Dtx_over_alph , lam1/alph);
    Dhprev=D*hprev;
    resid=(Dhprev-x);
    preact=hprev - Dt*(resid)./alph;
    %hcur = soft( hprev - Dt*(resid)./alph, lam1/alph);
    hcur = soft( preact , lam1/alph);

    result.Dhprev{niter}=Dhprev;
    result.resid{niter}=resid;
    result.preact{niter}=preact;
    result.hk{niter}=hcur;

    mse=f_mse(x,hcur);
    freg=f_reg(hcur);
    fcur=mse+freg;
    ftol_cur=(fprev-fcur)/fprev;
    
    if verbose
        fprintf('Iteration %d: fprev=%f, fcur=%f, mse=%f, reg=%f, rel. impr.=%f\n', niter, fprev, fcur, mse, freg, ftol_cur);
    end
    
    fprev=fcur;
    hprev=hcur;
    
    mse_all(niter+1)=mse;
    freg_all(niter+1)=freg;
    f_all(niter+1)=fcur;
    
    if abs(ftol)<1
        if ftol_cur<ftol
            done=1;
        end
    else
        if niter>=ftol
            done=1;
        end
    end
    
end

result.hstar=hprev;
result.f=f_all;
result.mse=mse_all;
result.reg=freg_all;
