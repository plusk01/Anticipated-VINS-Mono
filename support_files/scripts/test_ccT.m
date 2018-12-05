%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing the positive-definiteness of CC'
%
% See the Attention and Anticipation paper under equation (52).
%
% Parker Lusk
% 4 Dec 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;

% simulated parameters
sigma2 = 0.01;    % accel noise
sigmaba2 = 0.0001;  % accel bias
delta = 0.005;

% setup indices -- j > k
k = 10;
j = k+2;

nrImuMeasurements = j - k;

% calculate block coefficients: [a b; c d]
syms i;
CCt_11 = double(symsum( (j-i-0.5).^2 ,[k j-1]));
CCt_12 = double(symsum( (j-i-0.5)    ,[k j-1]));
a = CCt_11 * delta.^4 * sigma2;
b = CCt_12 * delta.^3 * sigma2;
d = (j-k-0) * delta.^2 * sigma2;

% I don't think the following is necessary, but it does change the
% eigenvalues and it makes this code numerically match Luca's. Could be an
% error in the C++?
a = nrImuMeasurements*a;

% Calculate CC'
CCT = [a*eye(3) b*eye(3); b*eye(3) d*eye(3)];

format('shortg');
eig(CCT)'
% det(CCT)
% cond(CCT)