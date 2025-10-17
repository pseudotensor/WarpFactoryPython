% Test WarpFactory MATLAB code
cd /WarpFactory
addpath(genpath('.'))

disp('=== Testing WarpFactory MATLAB Code ===');

% Test simple metric
disp('Creating Minkowski metric...');
metric = metricGet_Minkowski([5, 10, 10, 10]);
disp('Minkowski metric created');
disp(['Metric size: ', num2str(size(metric.tensor{1,1}))]);

% Test warp shell with paper parameters
disp('Creating Warp shell metric...');
metric_shell = metricGet_WarpShellComoving([1, 21, 21, 21], [0.5, 10.5, 10.5, 10.5], 4.49e27, 10, 20);
disp('Warp shell created');
disp(['Warp shell metric size: ', num2str(size(metric_shell.tensor{1,1}))]);

disp('=== WarpFactory test completed successfully ===');
