clc
clear all
%% load the data
load walk_left_featureVector
%% test an array of 3D points
npts = 47;
testpoints = zeros(npts, 3);
for i = 1:npts;
    testpoints(i, :) = data(1, i, 10, :);
end
%% plot 3D curve
testpoints = testpoints';
plot3(testpoints(1,:), testpoints(2,:), testpoints(3,:), 'ro', 'LineWidth',2);
text(testpoints(1,:), testpoints(2,:), testpoints(3,:), [repmat('  ',npts,1), num2str((1:npts)')])
ax = gca;
ax.XTick = [];
ax.YTick = [];
ax.ZTick = [];
box on
%% fit 3d points by a bspline
hold on
sp = cscvn(testpoints);
fnplt(sp,'r',2)
hold off
