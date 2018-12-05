%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Matrices for the Linear IMU Factor
%
% A transcription of Luca's C++ attention and anticipation code.
%
% Things changed for closer analysis (read: don't implement this way):
%   - Pose creation: I corced the frame rate of the camera to allow n IMUs
%   - Pose creation: I removed noise on timestep tj
%
% Parker Lusk
% 4 Dec 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;

%% Simulation parameters

imuDeltaT = 0.005; % sampling period of IMU
accVarianceDiscTime = 0.01;
biasAccVarianceDiscTime = 0.0001; %this depends on the IMU bias random walk
integrationVar = 1; % ??

%% Create pose data at frame i and j

t = 0.05; % frame rate of camera
t = imuDeltaT*2;

Ri = expm(skew([0 0 0]));
ti = 1;
Rj = expm(skew([1 0 1])*t);
tj = ti + t; %(t + 0.002*randn); % noisely create next timestep w/ std dev 2ms

%% Linearly interpolate IMU measurements between frame i and j

% Calculate the number of IMU measurements
Deltaij = tj - ti;
nrImuMeasurements = round(Deltaij/imuDeltaT)

% Since we rounded, there is some clock skew -- recalculate IMU rate
imuRate_ij = Deltaij / nrImuMeasurements;

% Sanity checks:
if (abs(imuRate_ij/imuDeltaT)>2)
    disp('imuRate_ij too different from imuDeltaT (too slow)');
end
if (abs(nrImuMeasurements*imuRate_ij - Deltaij)>1e-4)
    % not sure why this would ever happen...
    disp('Deltaij inconsistent with imuRate_ij');
end

Rij = Ri' * Rj; % relative rotation error
rotVec_ij = vex(logm(Rij)); % logmap to get to tangent space
Rimu = expm(skew(rotVec_ij / nrImuMeasurements)); % linear interpolation

%% Integrate "measurements" to obtain Aij and imuCovij

Nij = zeros(3,3); % eq (47)
Mij = zeros(3,3); % eq (48)

% initialize block coefficients
CCt_11 = 0;
CCt_12 = 0;

% "known" rotation at each IMU-rate timestep
Rh = Ri;

% note: w.r.t. draft: k=0 and j-1=nrImuMeasurements
for h = 0:(nrImuMeasurements-1) % the -1 is from the for loop < vs <=
    Mij = Mij + Rh;
    
    jkh = (nrImuMeasurements - h - 0.5);
    Nij = Nij + jkh * Rh;
    
    % entries of CCt
    CCt_11 = CCt_11 + jkh^2;
    CCt_12 = CCt_12 + jkh;
    
    % propagate rotation forward using incremental gyro measurement
    Rh = Rh * Rimu;
end

Nij = Nij * imuRate_ij^2;
Mij = Mij * imuRate_ij;

% Build cov(eta^imu_ij)
covImu = zeros(9,9);
covImu(1:3, 1:3) = eye(3) * (nrImuMeasurements * integrationVar * CCt_11 * imuRate_ij^4 * accVarianceDiscTime);
covImu(1:3, 4:6) = eye(3) * CCt_12 * imuRate_ij^3 * accVarianceDiscTime;
covImu(4:6, 1:3) = covImu(1:3, 4:6).';
covImu(4:6, 4:6) = eye(3) * nrImuMeasurements * imuRate_ij^2 * accVarianceDiscTime;
covImu(7:9, 7:9) = eye(3) * nrImuMeasurements * biasAccVarianceDiscTime;

% sanity check
if (norm(Rj - Rh) > 1e-2)
    disp('Rh integration is inconsistent');
end

% Build Ablk
Ai = -eye(9);
Ai(1:3,4:6) = -eye(3) * Deltaij;
Ai(1:3, 7:9) = Nij;
Ai(4:6, 7:9) = Mij;

eig(covImu(1:6,1:6))'