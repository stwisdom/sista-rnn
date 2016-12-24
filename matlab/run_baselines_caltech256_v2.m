% clear all;
clear variables;

cdir=pwd;
addpath('L1-homotopy');
cd('L1-homotopy'); setup_path; cd(cdir);
addpath('SpaRSA_2.0');

% load RandomStates (same as in L1-homotopy/demo_KalmanRWT
rseed = 2013;
rand('state',rseed);
randn('state',rseed);

% maxNumCompThreads(8);

verbose=0;

% dowmsampling factor
R=4;
N=128;
wavelet_type='daub8';
% load up the data
%data_name=sprintf('caltech256_R%d_N%d_%s',R,N,wavelet_type);
data_name='caltech256'
data=load(sprintf       ('../data_%s',data_name));

% remove training data, since we don't need it for these baselines:
data=rmfield(data,'xtrain');
data=rmfield(data,'ytrain');

xtest=data.xtest;
ytest=data.ytest;
PSI=data.D;
PHI=data.A;

[T,ntest,M]=size(xtest);
N=size(ytest,3);
PSI=PSI(1:N,1:N);

F=eye(N);

%lam1=0.02;
%lam2=0.002;
lam1=0.5;
lam2=1.0;

flag_save_ref=1;

% % have to break these out because parfor doesn't allow 3d matrices (why?!)
% iters_ista_all_fix_noOracle=zeros(T,ntest);
% iters_ista_all_conv_noOracle=zeros(T,ntest);
% iters_ista_all_fix_oracle=zeros(T,ntest);
% iters_ista_all_conv_oracle=zeros(T,ntest);
% iters_sparsa_all_fix_noOracle=zeros(T,ntest);
% iters_sparsa_all_conv_noOracle=zeros(T,ntest);
% iters_sparsa_all_fix_oracle=zeros(T,ntest);
% iters_sparsa_all_conv_oracle=zeros(T,ntest);
% iters_l1homotopy_all=zeros(T,ntest);
parfor itest=1:ntest
% for itest=1:ntest
    
    fprintf('Processing file %d of %d total\n',itest,ntest);
    
    cdir=pwd;
    addpath(genpath('L1-homotopy'));
%     cd('L1-homotopy'); setup_path; cd(cdir);
    addpath('SpaRSA_2.0');

    for tol=[3,1e-4]
    for flag_oracle_init=[0,1]
        
        ttic=tic;

        % target y for RNN is equal to sig for baselines
        sig=reshape(squeeze(ytest(:,itest,:)).',[T*N,1]);
        st=squeeze(ytest(:,itest,:));
        % input x for RNN is equal to noisy compressed measurements yt for
        % baselines
        yt=squeeze(xtest(:,itest,:));

        % parameters for ISTA
        ydiv=1; %division of 3 already done in python code
        Ast=((PHI./ydiv)*(st.')).';
        sigma=std(yt(:)./ydiv-Ast(:));
        q=max(abs(((PHI./ydiv)*PSI)'*(yt.')),[],1);
        alph=1;
        flag_debias=0;

        result_ista=cell(1,T);
        iters_ista=zeros(1,T);
        iters_sparsa=zeros(1,T);
        
        shat_ista=zeros(N,N);
        shat_sparsa=zeros(N,N);
        %shat_l1homotopy=zeros(N,N);
        tt=1;
        if flag_oracle_init
            s=squeeze(ytest(1,itest,:));
            hprev=(PSI')*s;
            hprev_sparsa=(PSI')*s;

            y=yt(tt,:).';
            s=st(tt,:).';
%             fprintf('t=%d, lam1=%f\n',tt,lam1);
            ybar=[y./ydiv; -sqrt(lam2).*F*PSI*hprev];
            Dbar=[(PHI./ydiv)*PSI; -sqrt(lam2).*PSI];
            cd(cdir);
            result_ista{tt}=ista(ybar,struct('lam1',lam1,'alph',alph,'D',Dbar,'ftol',tol,'hinit',(PSI.')*F*PSI*hprev,'verbose',0));
            hprev=result_ista{tt}.hstar;
            shat_ista(:,tt)=PSI*result_ista{tt}.hstar;
%             mse_ista(tt)=sum((PSI*result_ista{tt}.hstar - s).^2);
%             ser_ista(tt)=10*log10(sum(s.^2)/mse_ista(tt));
%             f_ista(tt)=result_ista{tt}.f(end);
            iters_ista(tt)=length(result_ista{tt}.f);

            [h,h_debias,obj]=SpaRSA(ybar,Dbar,lam1,'StopCriterion',1,'ToleranceA',tol,'Initialization',F*PSI*hprev_sparsa,'Debias',flag_debias,'VERBOSE',0);
%             mse_sparsa(tt)=sum((PSI*h - s).^2);
%             ser_sparsa(tt)=10*log10(sum(s.^2)/mse_sparsa(tt));
%             if ~isempty(h_debias)
%                 ser_sparsa_debias(tt)=10*log10(sum(s.^2)/sum((PSI*h_debias - s).^2));
%             end
%             f_sparsa(tt)=obj(end);
            iters_sparsa(tt)=length(obj);
            hprev_sparsa=h;
            shat_sparsa(:,tt)=PSI*h;
            
            %fprintf('ISTA:   SER=%.2fdB, MSE=%.3f\n',ser_ista(tt),mse_ista(tt));
            %fprintf('SpaRSA: SER=%.2fdB, MSE=%.3f\n',ser_sparsa(tt),mse_sparsa(tt));
        else
            % run ISTA for initial time step
            y=yt(1,:).';
            s=st(1,:).';
            cd(cdir);
            result_ista{1}=ista(y./ydiv,struct('lam1',lam1,'alph',alph,'D',(PHI./ydiv)*PSI,'ftol',tol,'verbose',0));
            shat_ista(:,tt)=PSI*result_ista{tt}.hstar;
%             mse_ista(1)=sum((PSI*result_ista{1}.hstar - s).^2);
%             ser_ista(1)=10*log10(sum(s.^2)/mse_ista(1));
%             f_ista(1)=result_ista{1}.f(end);
            iters_ista(1)=length(result_ista{1}.f);

            % run SpaRSA for initial time step
            [h,h_debias,obj]=SpaRSA(y./ydiv,(PHI./ydiv)*PSI,lam1,'StopCriterion',1,'ToleranceA',tol,'Debias',flag_debias,'VERBOSE',0);
%             mse_sparsa(1)=sum((PSI*h - s).^2);
%             ser_sparsa(1)=10*log10(sum(s.^2)/mse_sparsa(1));
%             if ~isempty(h_debias)
%                 ser_sparsa_debias(1)=10*log10(sum(s.^2)/sum((PSI*h_debias - s).^2));
%             end
%             f_sparsa(1)=obj(end);
            iters_sparsa(1)=length(obj);
            shat_sparsa(:,tt)=PSI*h;

            % report results
            %fprintf('ISTA:   SER=%.2fdB, MSE=%.3f, niters=%d\n',ser_ista(1),mse_ista(1),iters_ista(1));
            %fprintf('SpaRSA: SER=%.2fdB, MSE=%.3f, niters=%d\n',ser_sparsa(1),mse_sparsa(1),iters_sparsa(1));

            hprev=result_ista{1}.hstar;
            hprev_sparsa=h;
        end

        for tt=2:T
            y=yt(tt,:).';
            s=st(tt,:).';
            %lam1=max(1e-2*max(abs(((PHI./ydiv)*PSI)'*y)),sigma*sqrt(log(N)));
            if verbose
                fprintf('t=%d, lam1=%f\n',tt,lam1);
            end
            ybar=[y./ydiv; -sqrt(lam2).*F*PSI*hprev];
            Dbar=[(PHI./ydiv)*PSI; -sqrt(lam2).*PSI];
%             cd(cdir);
            result_ista{tt}=ista(ybar,struct('lam1',lam1,'alph',alph,'D',Dbar,'ftol',tol,'hinit',(PSI.')*F*PSI*hprev,'verbose',0));
            hprev=result_ista{tt}.hstar;
            shat_ista(:,tt)=PSI*result_ista{tt}.hstar;
%             mse_ista(tt)=sum((PSI*result_ista{tt}.hstar - s).^2);
%             ser_ista(tt)=10*log10(sum(s.^2)/mse_ista(tt));
%             f_ista(tt)=result_ista{tt}.f(end);
            iters_ista(tt)=length(result_ista{tt}.f);

            [h,h_debias,obj]=SpaRSA(ybar,Dbar,lam1,'StopCriterion',1,'ToleranceA',tol,'Initialization',F*PSI*hprev_sparsa,'Debias',flag_debias,'VERBOSE',0);
%             mse_sparsa(tt)=sum((PSI*h - s).^2);
%             ser_sparsa(tt)=10*log10(sum(s.^2)/mse_sparsa(tt));
%             if ~isempty(h_debias)
%                 ser_sparsa_debias(tt)=10*log10(sum(s.^2)/sum((PSI*h_debias - s).^2));
%             end
%             f_sparsa(tt)=obj(end);
            iters_sparsa(tt)=length(obj);
            hprev_sparsa=h;
            shat_sparsa(:,tt)=PSI*h;

            %fprintf('ISTA:   SER=%.2fdB, MSE=%.3f\n',ser_ista(tt),mse_ista(tt));
            %fprintf('SpaRSA: SER=%.2fdB, MSE=%.3f\n',ser_sparsa(tt),mse_sparsa(tt));

        end %for t=2:T
        
%         psnr_ista=10*log10(max(sig.^2)/(mean(mse_ista)/N));
%         psnr_sparsa=10*log10(max(sig.^2)/(mean(mse_sparsa)/N));
%         
%         fprintf('ISTA:   SER=%.2fdB, PSNR=%.2fdB, MSE=%.3f\n',mean(ser_ista),mean(psnr_ista),mean(mse_ista)/N);
%         fprintf('SpaRSA: SER=%.2fdB, PSNR=%.2fdB, MSE=%.3f\n',mean(ser_sparsa),mean(psnr_sparsa),mean(mse_sparsa)/N);

        if (tol<1) && flag_oracle_init
            % run \ell_1-homotopy by [Asif and Romberg 2014]:
            results=apply_baselines_caltech256(sig,yt./ydiv,struct('R',R,'N',N,'P',3,'PHI',PHI./ydiv,'flag_plot',0));
            cd(cdir);
            shat_l1homotopy=reshape(results.sigh_vec(:,1),[N,N]);
%             mses_l1homotopy(itest,:)=results.mse./(N*(T-2));
%             sers_l1homotopy(itest,:)=results.ser;
%             psnrs_l1homotopy(itest,:)=10*log10(max(sig.^2)/mses_l1homotopy(itest,:));
            iters_l1homotopy=results.iters;
        else
            shat_l1homotopy=[];
%             mses_l1homotopy=[];
%             sers_l1homotopy=[];
%             psnrs_l1homotopy=[];
%             iters_l1homotopy=[];
        end
        
        toc(ttic);
        
        %undo normalization of images:
        % python code is as follows:
%       #write out images from initial network parameters
%       # undo scaling and demeaning for the estimated images:
%       yest_test_init = yest_test_init/config['std_cols']
%       yest_test_init = yest_test_init*mean_ystd
%       yest_test_init = yest_test_init+mean_ymean
%       # clip yest to be between 0.0 and 1.0
%       yest_test_init[yest_test_init<0.0]=np.float32(0.0)
%       yest_test_init[yest_test_init>1.0]=np.float32(1.0)
        s_ref=undo_norm(squeeze(ytest(:,itest,:)).',data.std_cols,data.mean_ystd,data.mean_ymean);
        shat_ista=undo_norm(shat_ista,data.std_cols,data.mean_ystd,data.mean_ymean);
        shat_sparsa=undo_norm(shat_sparsa,data.std_cols,data.mean_ystd,data.mean_ymean);
        
        if flag_oracle_init
            string_oracle='oracle';
        else
            string_oracle='noOracle';
        end
        savename=sprintf('baseline_%s_K%d_lam1%f_lam2%f_%s',string_oracle,tol,lam1,lam2,data_name);
        path_root=fullfile('../caltech256',savename);
        path_ref=fullfile('../caltech256/ref_matlab');
        savefile=strsplit(strtrim(data.yfiles_test(itest,:)),'caltech256/');
        savefile=savefile{end};
        savefile=strrep(savefile,'.jpg','.png');
        
        % write out files
        path_ista=fullfile(path_root,'recon_ista');
        file_ista=fullfile(path_ista,savefile);
        if isempty(dir(fileparts(file_ista)))
            mkdir(fileparts(file_ista));
        end
        imwrite(shat_ista,file_ista);
        iters=iters_ista;
        dlmwrite(strrep(file_ista,'.png','.iters'),iters);
        
        path_sparsa=fullfile(path_root,'recon_sparsa');
        file_sparsa=fullfile(path_sparsa,savefile);
        if isempty(dir(fileparts(file_sparsa)))
            mkdir(fileparts(file_sparsa));
        end
        imwrite(shat_sparsa,file_sparsa);
        iters=iters_sparsa;
        dlmwrite(strrep(file_sparsa,'.png','.iters'),iters);
        
        if ~isempty(shat_l1homotopy)
            path_l1homotopy=fullfile(path_root,'recon_l1homotopy');
            file_l1homotopy=fullfile(path_l1homotopy,savefile);
            if isempty(dir(fileparts(file_l1homotopy)))
                mkdir(fileparts(file_l1homotopy));
            end
            shat_l1homotopy=undo_norm(shat_l1homotopy,data.std_cols,data.mean_ystd,data.mean_ymean);
            imwrite(shat_l1homotopy,file_l1homotopy);
            iters=iters_l1homotopy;
            dlmwrite(strrep(file_l1homotopy,'.png','.iters'),iters);
            
            if flag_save_ref
                file_ref=fullfile(path_ref,savefile);
                if isempty(dir(fileparts(file_ref)))
                    mkdir(fileparts(file_ref));
                end
                imwrite(s_ref,file_ref);
            end
            
        end
        
    end %for flag_oracle_init
    end %for tol
end %parfor itest
