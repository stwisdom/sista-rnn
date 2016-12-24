clear variables;

path_ref='../caltech256/ref_matlab';
path_root='../caltech256';
dirs_exp={
    'K3_lr0-0001_lam0-5_v3_ds1_N128_latticeNoInputConn_randomInit/recon_best',...
    'K3_lr0-0001_LSTM_ds1_N128_100epochs/recon_best',...
    'K3_lr0-0001_lam0-5_v3_ds1_N128_100epochs_trainsista/recon_best',...
};

labels_exp={
    'RNN_randInit_ds1',...
    'LSTM_randInit_ds1',...
    'SISTA-RNN_SISTAparams_ds1_lam1_0-5',...
};

%results_mat='scores_caltech256_all.mat';

results_mat='../scores_caltech256_LSTM_SISTAparams.mat';
dirs_exp=dirs_exp(end-2:end);
labels_exp=labels_exp(end-2:end);

nexp=length(dirs_exp);


% dowmsampling factor
R=4;
N=128;
wavelet_type='daub8';
% load up the data
data=load('../data_caltech256.mat');

yfiles_test=data.yfiles_test;

nfiles=size(yfiles_test,1);
files_image=cell(nfiles,1);
for ifile=1:nfiles
    file_cur=strsplit(strtrim(yfiles_test(ifile,:)),'caltech256/');
    files_image{ifile}=strrep(file_cur{end},'.jpg','.png');
end

scores=zeros(nexp,nfiles,5);
tic;
for ifile=1:nfiles

    if mod(ifile,floor(nfiles/100))==0
        fprintf('Scored %d files of %d total for %d experiments each\n',ifile,nfiles,nexp);
        toc;
        tic;
    end

    ref=imread(fullfile(path_ref,files_image{ifile}));
    %tic;
    for iexp=1:nexp
        path_image=fullfile(path_root,dirs_exp{iexp},files_image{ifile});
        A=imread(path_image);
        scores(iexp,ifile,1:4)=score_image(A,ref);
        path_iters=strrep(path_image,'.png','.iters');
        if exist(path_iters,'file')
            iters=dlmread(path_iters);
            scores(iexp,ifile,5)=max(iters);
        else
            scores(iexp,ifile,5)=3;
        end
    end
    %toc;
end

labels_scores={'MSE','PSNR','SNR','SSIM','IterMax'};
save(results_mat,'scores','labels_exp','dirs_exp','labels_scores');
