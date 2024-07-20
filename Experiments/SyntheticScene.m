clear; close all;

%% Add functions to working dir
addpath(genpath(pwd));

%% View synthetic 3D scene
[calMatrices, R_t0, matchingPoints, points3D] = GenerateSyntheticScene(3010, 1, 30, 50, 0);
scatter3(points3D(1,:), points3D(2,:), points3D(3,:))

%% Save synthetic 3D scenes w.r.t. simulation iteration
saveas(gcf, strcat('Experiments/Synthetic/scene/IT30.png'), 'png');
