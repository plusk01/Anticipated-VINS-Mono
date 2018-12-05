%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot cost of feature tracker
%
% Parker Lusk
% 27 Nov 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;

plotCosts(1, 'run2_afs_cost.bin', 'AFS Costs for EuRoC MH 01 Easy');
plotCosts(2, 'run2_orig_cost.bin', 'Original Costs for EuRoC MH 01 Easy');

function costs = plotCosts(fignum, file, figtitle)

    real_time_line = 33;

    % Read the binary log with doubles
    log = fopen(file);
    costs = fread(log, 'double');

    figure(fignum), clf; hold on;
    plot(costs)
    plot(smoothdata(costs,'gaussian',50), 'LineWidth',3)
    title(figtitle);
    xlabel('frame'); ylabel('cost [ms]');
    hline = refline([0 real_time_line]); hline.Color = 'k';hline.LineWidth = 3;
    hline2=refline([0 mean(costs)]);hline2.Color='k';hline2.LineWidth=1;
    
    xaxis([0 length(costs)]);
    yaxis([0 45]);
    
    disp(figtitle)
    mean(costs)
    sum(costs>real_time_line)

end