function [st,yt,PSI,PHI]=generate_KalmanRWT_signals_v3(R,PHI,N,P)
% function [st,yt,PSI,PHI]=generate_KalmanRWT_signals_v3(R,PHI,N,P)
% Generates a pair of synthetic sequences x and y that are 
% 128 x 256 and 128 x 256/R, respectively.
%
% Code taken from:
% demo_KalmanRWT
%
% Solves the following dynamic BPDN problem over a window t = t_1,...,t_L
% min_x \sum_t \|W_t x_t\|_1 + 1/2*||A_t x_t - y_t||_2^2 + 1/2||F_t x_t - x_t+1\|_2^2
%
% which updates the solution as the signal changes according to a linear
% dynamical system.
%
% for instance, y_t = A_t x_t + e_t
%               x_t+1 = F_t x_t + f_t 
%       where F_t is a partially known function that models prediction
%       between the consecutive x_t and f_t denotes the prediction error 
%       (e.g., a random drift)
%
% Applications:
%       streaming signal recovery using a dynamic model
% 
%   	track a signal as y, A, and/or x change... 
%       predict an estimate of the solution and
%       update weights according to the predicted solution
%
% Written by: Salman Asif, Georgia Tech
% Email: sasif@gatech.edu
% Created: November 2012

% clear
% close all force

% % Limit the number of computational threads (for profiling)
% maxNumCompThreads(1);

if ~exist('PHI','var')
    PHI=[];
end

%% Setup path
cdir=pwd;
mname = mfilename;
mpath = mfilename('fullpath');
mdir = mpath(1:end-length(mname));
cd(mdir);

addpath utils/
addpath utils/utils_Wavelet
addpath utils/utils_LOT
addpath solvers/

% fprintf(['----------',datestr(now),'-------%s------------------\n'],mname)

% % load RandomStates
% %
% rseed = 2013;
% rand('state',rseed);
% randn('state',rseed);

% simulation parameters
mType = 'sign'; % {'randn','orth','rdct'};
mFixed = 1; % measurement system time-(in)variant
sType = 'pcwreg'; % {'heavisine', 'pcwreg', 'blocks','pcwPoly'}
SNR = 35;       % additive Gaussian noise

wt_pred = sqrt(0.5);

if ~exist('N','var')
    N = 256;   % signal length
end
% R = 4; % compression rate
M = round(N/R);    % no. of measurements

LM = 1*N; % LM: length of measurement window
LS_Kalman = 'smooth'; % {'filter','smooth','inst'};

% streaming window
if ~exist('P','var')
    P = 3; % size of the working window is P*N
end

% signal length
sig_length = 2^15; % 128*128;

% signal dynamics
dType = 'crshift'; % type of dynamics 'crshift', or 'static'
cshift = -1;
rshift_max = 0.5;
rshift_h = @(z) (rand-0.5)*rshift_max*2;

% DWT parameters
% type of scaling function
% depth of scaling functions (number of wavelet levels)
% type of extension for DWT (0 - periodic extension, 3 - streaming)
wType = 'daub79'; sym = 1;
wType = 'daub8'; sym = 0;
J = log2(N)-3;

% rank-1 update mode
delx_mode = 'mil'; % mil or qr

% add snapshots of the signal in streaming window and average them before comitting to the output.
avg_output = 0; 

verbose = 0;


%% SM: Sampling modes
% % LM: length of measurement window
% LM = 2*N; % 'universal' sampling scheme (align before the overlapping regions of DWT windows that are measured)
if LM > N
    LeftEdge_trunc = 1;
else
    LeftEdge_trunc = 0;
end
LeftProj_cancel = 1;


%%
% fprintf('CS-Kalman tracking a dynamical signal and reweighting..\n');
% str0 = sprintf('mType-%s, sType-%s, SNR = %d, (N,M,R) = %d, %d, %d, P = %d, LM = %d, LS_Kalman-%s \n wType-%s, J = %d, sym = %d, specified signal-length = %d, \n dType-%s, cshift = %d, rshift_max = %0.3g, wt_pred = %0.3g. ', mType, sType, SNR, N, round(N/R), R, P, LM, LS_Kalman, wType, J, sym, sig_length, dType, cshift, rshift_max, wt_pred);
% disp(str0);

%% DWT setup
% DWT parameters
% Length of each window is L. (extend to adaptive/dyadic windows later?)
% wType = 'daub4'; % type of scaling function
% J = 3; % depth of scaling functions (number of wavelet levels)
% sym = 3; % type of extension for DWT (0 - periodic extension, 3 - streaming)
in_Psi = []; in_Psi.N = N; in_Psi.J = J; in_Psi.wType = wType; in_Psi.sym = sym;
Psi = create_DWT(in_Psi); % DWT synthesis matrix over a window
L = size(Psi,1);

%% Signal generation

% Setup dynamical model
% At every time instance, add to the original/previous signal
% an integer circular shift that is known
% a random drift that is unknown
if strcmpi(dType, 'crshift')
    % Generate a signal by circular shift and a random drift in a seed
    % signal
    in = []; in.type = sType; in.randgen = 0; in.take_fwt = 0;
    [x_init sig wave_struct] = genSignal(N,in);
    
    F_h = @(x,cshift,rshift) interp1(1:N,circshift(x,cshift),[1:N]+rshift,'linear','extrap')';
    
    F0 = zeros(N);
    for ii = 1:N;
        F0(:,ii) = F_h(circshift([1; zeros(N-1,1)],ii-1),cshift,0);
    end
    sigt = sig; sig = [];
    for ii = 1:round(sig_length/N);
        rshift = rshift_h(1);
        sigt = F_h(sigt, cshift, rshift);
        sig = [sig; sigt];
    end
else
    % Generate a predefined streaming signal
    in = []; in.type = sType; in.randgen = 0; in.take_fwt = 0;
    [x_init sig wave_struct] = genSignal(N,in);
    
    cshift = 0; rshift = 0;
    F_h = @(x,cshift,rshift) x;
    F0 = eye(N);
end
% sig = [zeros(L-N,1);sig];
sig_length = length(sig);

% view DWT coefficients...
alpha_vec = apply_DWT(sig,N,wType,J,sym);
% figure(123);
% subplot(211); imagesc(reshape(alpha_vec,N,length(alpha_vec)/N));
% axis xy;
% subplot(212); plot(alpha_vec);

% view innovations in the signal.. 
% dsig = []; for n = 0:N:length(sig)-N; dsig = [dsig; sig(n+1:n+N)-circshift(sig(n+N+1:n+2*N),1)]; figure(1); plot([sig(n+1:n+N) sig(n+1:n+N)-circshift(sig(n+N+1:n+2*N),1)]); pause; end

% Simulation parameters
streaming_iter = ceil(length(sig)/N);
SIM_stack = cell(streaming_iter,1);
SIM_memory = cell(streaming_iter,1);

x_vec = zeros(N*streaming_iter,1);
xh_vec = zeros(N*streaming_iter,3);
sig_vec = zeros(length(sig),1);
sigh_vec = zeros(length(sig),3);

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

%% Dynamics matrix
F = zeros(P*N,(P+1)*N);
for p = 1:P
    F((p-1)*N+1:p*N,(p-1)*N+1:(p+1)*N) = [F0 -eye(N)];
end
F = wt_pred*F(:,N+1:end);

%% Create analysis/synthesis matrix explicitly and compute sparse coeffs.
in = [];
in.P = P; in.Psi = Psi;
% in.P = P; in.Jp = Jp; in.wType = wType; in.N = N; in.sym = sym;
PSI = create_PSI_DWT(in);

% Sparse coefficients...
T_length = size(PSI,1);
t_ind = 1:T_length;

sigt = sig(t_ind); % Signal under the LOT window at time t
if sym == 1 || sym == 2
    x = pinv(PSI'*PSI)*(PSI'*sigt); % Sparse LOT coefficients
else
    x = PSI'*sigt;
end

%% initialize with a predicted value of first x
% xt = x(1:N);
%
% At = genAmat_h(M,N);
% sigma = sqrt(norm(At*xt)^2/10^(SNR/10)/M);
% e = randn(M,1)*sigma;
% yt = At*xt+e;
%
% tau = max(1e-2*max(abs(At'*yt)),sigma*sqrt(log(N)));
%
% % rwt L1 with the first set of measurement...
% in = [];
% in.tau = tau; W = tau;
% in.delx_mode = delx_mode;
% for wt_itr = 1:5
%     W_old = W;
%
%     out = l1homotopy(At,yt,in);
%     xh = out.x_out;
%
%     % Update weights
%     xh_old = xh;
%     alpha = 1; epsilon = 1;
%     beta = M*(norm(xh_old,2)/norm(xh_old,1))^2;
%     W = tau/alpha./(beta*abs(xh_old)+epsilon);
%
%     yh = At*xh_old;
%     Atr = At'*(At*xh-yt);
%     u =  -W.*sign(xh)-Atr;
%     pk_old = Atr+u;
%
%     in = out;
%     in.xh_old = xh;
%     in.pk_old = pk_old;
%     in.u = u;
%     in.W_old = W_old;
%     in.W = W;
% end
% xh(abs(xh)<tau/sqrt(log(N))) = 0;


% Another way to initialize...
% Best M/2-sparse signal...
% [val_sort ind_sort] = sort(abs(x),'descend');
% xh = x;
% xh(ind_sort(P*N/2+1:end)) = 0;

% Oracle value for the initialization
% xh = x; disp('oracle initialization');

% model for the outgoing window...
% sim = 1;
% st_ind = N;
% t_ind = st_ind+t_ind;
% s_ind = t_ind(1:L);
% 
% sig_out = PSI(st_ind+1:st_ind+N,:)*xh;
% xh = xh(st_ind+1:end);
% 
% xh_out = xh(1:N);
% x_vec((sim-1)*N+1:sim*N,1) = x(st_ind+1:st_ind+N);
% xh_vec((sim-1)*N+1:sim*N,1:3) = [xh_out xh_out xh_out];
% 
% sig_temp = Psi*xh_out;
% sig_temp = [sig_out; sig_temp(N+1:end)];
% sig_vec(s_ind) = sigt(s_ind);
% sigh_vec(s_ind,1:3) = sigh_vec(s_ind,1:3)+[sig_temp sig_temp sig_temp];


%% Generate complete measurement system
% Sparse coefficients...
t_ind = t_ind + N;
sigt = sig(t_ind); % Signal under the LOT window at time t
if sym == 1 || sym == 2
    x = pinv(PSI'*PSI)*(PSI'*sigt); % Sparse LOT coefficients
else
    x = PSI'*sigt;
end

y = PHI*sigt(1:end-(L-N));

leny = length(y);
sigma = sqrt(norm(y)^2/10^(SNR/10)/leny);
e = randn(leny,1)*sigma;
y = y+e;


PSI_M = PSI(1:end-(L-N),:);
A = [PHI; F]*PSI_M;


% sig_out = sigh_vec(t_ind(1:N)-N,1);
% y = [y; -wt_pred*F0*sig_out; zeros((P-1)*N,1)];
% 
% % REMOVE the part of outgoing DWT projection in the overlapping region
% % on left side of streaming window...
% if LeftProj_cancel
%     y = y-[PHI(:,1:(L-N));F(:,1:(L-N))]*(Psi(end-(L-N)+1:end,:)*xh_out(1:N));
% end

% %% parameter selection 
% % tau = sigma*sqrt(log(N));
% tau = max(1e-2*max(abs(A'*y)),sigma*sqrt(log(P*N)));
% 
% maxiter = 2*P*N;
% err_fun = @(z) (norm(x-z)/norm(x))^2;

%% GO...

T=sig_length/N;
st=zeros(N,T);
yt=zeros(M,T);
PHI=genAmat_h(M,N);

for tidx=1:T
    
    st(:,tidx)=sig((tidx-1)*N+1:tidx*N);
    yt(:,tidx)=PHI*st(:,tidx)+randn(M,1)*sigma;
    
end

st=st.'; %now TxN
yt=yt.'; %now TxM

cd(cdir);
