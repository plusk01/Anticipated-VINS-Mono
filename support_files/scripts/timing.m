%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Timing analysis of:
%   - feature tracker
%   - feature selector
%   - estimator/optimizer
%
% Parker Lusk
% 13 Dec 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;

plotCosts(1, '~/.ros/afs_cost.bin', 'Feature Detector');
plotCosts(2, '~/.ros/fsel_cost.bin', 'Feature Selector');
plotCosts(3, '~/.ros/est_cost.bin', 'Estimator');

function costs = plotCosts(fignum, file, figtitle)

    real_time_line = 33;

    % Read the binary log with doubles
    log = fopen(file);
    costs = fread(log, 'double');

    figure(fignum), clf; hold on;
    plot(costs)
    plot(smoothdata(costs,'gaussian',50), 'LineWidth',3)
    title(figtitle);
    xlabel('frame'); ylabel('time [ms]');
    hline = refline([0 real_time_line]); hline.Color = 'k';hline.LineWidth = 3;
    hline2=refline([0 mean(costs)]);hline2.Color='k';hline2.LineWidth=1;
    
    xaxis([0 length(costs)]);
    yaxis([0 45]);
    
    disp(figtitle)
    mean(costs)
    sum(costs>real_time_line)

end