clear variables;

L=load('../scores_caltech256_LSTM_SISTAparams.mat');

scores=L.scores;
labels_exp=L.labels_exp;
nexp=size(scores,1);

ItersMax=squeeze(max(scores(:,:,5),[],2));
MSE=squeeze(mean(scores(:,:,1),2));
RMSE=sqrt(MSE);
PSNR=squeeze(mean(scores(:,:,2),2));
SSIM=squeeze(mean(scores(:,:,4),2));
T=table(ItersMax,MSE,PSNR,SSIM,'RowNames',labels_exp)

