function res = int_spline(sp1, sp2)
% INT_SPLINE computes the integral of multiplication of two splines which 
% have the same knots
% sp1, sp2 ... two spline objects
% result ... scale of integral
    if length(sp1.breaks) ~= length(sp2.breaks)
        error('knots of two splines are not equal!')
    end
    breaks = sp1.breaks;
    grid = linspace(min(breaks), max(breaks), length(breaks)*10);
    samples1 = fnval(sp1, grid);
    samples2 = fnval(sp2, grid);
%     dist = 0;
%     for i = 1:length(grid);
%         dist = dist + norm(samples1(:,i) - samples2(:,i), 1);
%     end
%     dist = dist/length(grid);
    res = 0;
    for i = 1:length(grid);
        res = res + dot(samples1(:,1), samples2(:,i));
    end
    res = res/length(grid);
end