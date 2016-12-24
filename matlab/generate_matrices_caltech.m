%function [PSI,PHI,PSIinv]=generate_matrices_caltech256(R,N)

clear variables;
rseed = 2013;
rand('state',rseed);
randn('state',rseed);

R=4;
N=128;

%wavelet_type='waveletcdf97'
wavelet_type='daub8'

PSI=[];
PHI=[];

%% Setup path               
cdir=pwd;                       
cd('L1-homotopy')

addpath utils/                                      
addpath utils/utils_Wavelet                         
addpath utils/utils_LOT                             
addpath solvers/

% simulation parameters
mType = 'sign'; % {'randn','orth','rdct'};
mFixed = 1; % measurement system time-(in)variant
% N = 256;   % signal length
% R = 4; % compression rate
M = round(N/R);    % no. of measurements
LM = 1*N; % LM: length of measurement window
P = 1;

%% Setup sensing matrices
in = []; in.type = mType;
if mFixed
    if ~isempty(PHI)
        At = PHI;
        in.PHI=PHI;
    else
        At = genAmat(M,LM,in);
    end
    genAmat_h = @(m,n) At;
else
    genAmat_h = @(M,N) genAmat(M,N,in);
end
in.P = P-(LM-N)/N;
in.LM = LM; in.M = M; in.N = N;
PHI = create_PHI(in);

cd(cdir);

% Setup inverse transform matrix
switch(wavelet_type)
    case 'waveletcdf97'
        [~,PSI]=waveletcdf97_matrix(N,-nextpow2(N));
        [~,PSIinv]=waveletcdf97_matrix(N,nextpow2(N));
    case 'daub8'
        cd('L1-homotopy')
        [~,~,PSI]=generate_KalmanRWT_signals_v3(R,[],N,1);
        PSIinv=PSI.';
        cd(cdir);
end

% Save off variables
D=PSI;
Dinv=PSIinv;
A=PHI;
F=eye(N);
if strmatch(wavelet_type,'waveletcdf97')
    save(sprintf('matrices_caltech_R%d_N%d.mat',R,N),'D','Dinv','A','F','-v7.3');
else
    save(sprintf('matrices_caltech_R%d_N%d_%s.mat',R,N,wavelet_type),'D','Dinv','A','F','-v7.3');

end
