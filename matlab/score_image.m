function [S,labels]=score_image(A,ref)
    S=zeros(1,4);
    
    %MSE
    S(1)=immse(A,ref);
    
    %PSNR and SNR
    [S(2),S(3)]=psnr(A,ref);
        
    %SSIM
    S(4)=ssim(A,ref);
    
    labels={'MSE','PSNR','SNR','SSIM'};
