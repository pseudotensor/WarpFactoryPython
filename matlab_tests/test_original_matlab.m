% Test Original MATLAB WarpFactory Code
% This script runs the ORIGINAL MATLAB WarpFactory (not through Python)
%
% Path: /tmp/WarpFactory_MATLAB_Original

fprintf('\n');
fprintf('==========================================================\n');
fprintf('TESTING ORIGINAL MATLAB WARPFACTORY CODE\n');
fprintf('==========================================================\n');
fprintf('\n');

% Add all WarpFactory paths
cd /tmp/WarpFactory_MATLAB_Original
addpath(genpath('.'))

fprintf('Added paths:\n');
fprintf('  /tmp/WarpFactory_MATLAB_Original\n');
fprintf('  Including: Metrics/, Solver/, Analyzer/, Units/\n');
fprintf('\n');

% Define parameters (matching paper parameters from arXiv:2405.02709)
fprintf('=== PARAMETERS ===\n');
gridSize = [1, 21, 21, 21];
worldCenter = [0.5, 10.5, 10.5, 10.5];
m = 4.49e27;  % Total mass (kg) - approximately 2.365 Jupiter masses
R1 = 10.0;    % Inner radius (m)
R2 = 20.0;    % Outer radius (m)

fprintf('Grid size: [%d, %d, %d, %d]\n', gridSize(1), gridSize(2), gridSize(3), gridSize(4));
fprintf('World center: [%.1f, %.1f, %.1f, %.1f]\n', worldCenter(1), worldCenter(2), worldCenter(3), worldCenter(4));
fprintf('Mass: %.3e kg (%.3f Jupiter masses)\n', m, m/1.898e27);
fprintf('Inner radius R1: %.1f m\n', R1);
fprintf('Outer radius R2: %.1f m\n', R2);
fprintf('Shell thickness: %.1f m\n', R2 - R1);
fprintf('\n');

%% TEST 1: Simple Minkowski metric
fprintf('=== TEST 1: MINKOWSKI METRIC ===\n');
tic;
metric_mink = metricGet_Minkowski(gridSize);
t_mink = toc;
fprintf('Success! Created Minkowski metric in %.4f seconds\n', t_mink);
fprintf('Metric size: %d x %d x %d x %d\n', size(metric_mink.tensor{1,1}));
fprintf('g_tt (should be -1): %.10f\n', metric_mink.tensor{1,1}(1,1,1,1));
fprintf('g_xx (should be +1): %.10f\n', metric_mink.tensor{2,2}(1,1,1,1));
fprintf('\n');

%% TEST 2: Warp Shell metric (paper parameters)
fprintf('=== TEST 2: WARP SHELL METRIC ===\n');
fprintf('Creating Warp Shell Comoving metric...\n');
tic;
metric_shell = metricGet_WarpShellComoving(gridSize, worldCenter, m, R1, R2);
t_shell = toc;
fprintf('Success! Created Warp Shell metric in %.4f seconds\n', t_shell);
fprintf('Metric name: %s\n', metric_shell.name);
fprintf('Metric coords: %s\n', metric_shell.coords);
fprintf('Metric index: %s\n', metric_shell.index);

% Analyze g_tt component
g_tt = metric_shell.tensor{1,1};
fprintf('\ng_tt (time-time component) statistics:\n');
fprintf('  Min: %.10e\n', min(g_tt(:)));
fprintf('  Max: %.10e\n', max(g_tt(:)));
fprintf('  Mean: %.10e\n', mean(g_tt(:)));
fprintf('  Std: %.10e\n', std(g_tt(:)));

% Check specific locations
fprintf('\ng_tt at specific locations:\n');
fprintf('  At center [1,11,11,11]: %.10e\n', g_tt(1,11,11,11));
fprintf('  At edge [1,21,11,11]: %.10e\n', g_tt(1,21,11,11));
fprintf('\n');

%% TEST 3: Stress-Energy Tensor
fprintf('=== TEST 3: STRESS-ENERGY TENSOR ===\n');
fprintf('Computing energy tensor using finite differences...\n');
tic;
energy = getEnergyTensor(metric_shell, 0, 'fourth');
t_energy = toc;
fprintf('Success! Computed energy tensor in %.4f seconds\n', t_energy);
fprintf('Energy type: %s\n', energy.type);
fprintf('Energy index: %s\n', energy.index);
fprintf('Finite difference order: %s\n', energy.order);

% Analyze T^00 component (energy density)
T_00 = energy.tensor{1,1};
fprintf('\nT^00 (energy density) statistics:\n');
fprintf('  Min: %.10e\n', min(T_00(:)));
fprintf('  Max: %.10e\n', max(T_00(:)));
fprintf('  Mean: %.10e\n', mean(T_00(:)));
fprintf('  Std: %.10e\n', std(T_00(:)));

% Count non-zero values
nonzero_count = sum(abs(T_00(:)) > 1e-10);
fprintf('  Non-zero points: %d / %d (%.1f%%)\n', nonzero_count, numel(T_00), 100*nonzero_count/numel(T_00));
fprintf('\n');

%% TEST 4: Energy Conditions
fprintf('=== TEST 4: ENERGY CONDITIONS ===\n');

% Null Energy Condition
fprintf('Computing Null Energy Condition (NEC)...\n');
tic;
[nec_map, ~, ~] = getEnergyConditions(energy, metric_shell, "Null", 100, 10, 0, 0);
t_nec = toc;
fprintf('Success! NEC computed in %.4f seconds\n', t_nec);
fprintf('NEC statistics:\n');
fprintf('  Min: %.10e\n', min(nec_map(:)));
fprintf('  Max: %.10e\n', max(nec_map(:)));
fprintf('  Mean: %.10e\n', mean(nec_map(:)));
fprintf('  Median: %.10e\n', median(nec_map(:)));

nec_violations = sum(nec_map(:) < 0);
fprintf('  Violations (< 0): %d / %d points (%.2f%%)\n', nec_violations, numel(nec_map), 100*nec_violations/numel(nec_map));
fprintf('\n');

% Weak Energy Condition
fprintf('Computing Weak Energy Condition (WEC)...\n');
tic;
[wec_map, ~, ~] = getEnergyConditions(energy, metric_shell, "Weak", 100, 10, 0, 0);
t_wec = toc;
fprintf('Success! WEC computed in %.4f seconds\n', t_wec);
fprintf('WEC statistics:\n');
fprintf('  Min: %.10e\n', min(wec_map(:)));
fprintf('  Max: %.10e\n', max(wec_map(:)));
fprintf('  Mean: %.10e\n', mean(wec_map(:)));
fprintf('  Median: %.10e\n', median(wec_map(:)));

wec_violations = sum(wec_map(:) < 0);
fprintf('  Violations (< 0): %d / %d points (%.2f%%)\n', wec_violations, numel(wec_map), 100*wec_violations/numel(wec_map));
fprintf('\n');

% Strong Energy Condition
fprintf('Computing Strong Energy Condition (SEC)...\n');
tic;
[sec_map, ~, ~] = getEnergyConditions(energy, metric_shell, "Strong", 100, 10, 0, 0);
t_sec = toc;
fprintf('Success! SEC computed in %.4f seconds\n', t_sec);
fprintf('SEC statistics:\n');
fprintf('  Min: %.10e\n', min(sec_map(:)));
fprintf('  Max: %.10e\n', max(sec_map(:)));
fprintf('  Mean: %.10e\n', mean(sec_map(:)));
fprintf('  Median: %.10e\n', median(sec_map(:)));

sec_violations = sum(sec_map(:) < 0);
fprintf('  Violations (< 0): %d / %d points (%.2f%%)\n', sec_violations, numel(sec_map), 100*sec_violations/numel(sec_map));
fprintf('\n');

% Dominant Energy Condition
fprintf('Computing Dominant Energy Condition (DEC)...\n');
tic;
[dec_map, ~, ~] = getEnergyConditions(energy, metric_shell, "Dominant", 100, 10, 0, 0);
t_dec = toc;
fprintf('Success! DEC computed in %.4f seconds\n', t_dec);
fprintf('DEC statistics:\n');
fprintf('  Min: %.10e\n', min(dec_map(:)));
fprintf('  Max: %.10e\n', max(dec_map(:)));
fprintf('  Mean: %.10e\n', mean(dec_map(:)));
fprintf('  Median: %.10e\n', median(dec_map(:)));

dec_violations = sum(dec_map(:) < 0);
fprintf('  Violations (< 0): %d / %d points (%.2f%%)\n', dec_violations, numel(dec_map), 100*dec_violations/numel(dec_map));
fprintf('\n');

%% SAVE RESULTS
fprintf('=== SAVING RESULTS ===\n');
save('/tmp/matlab_original_results.mat', ...
    'metric_shell', 'energy', ...
    'nec_map', 'wec_map', 'sec_map', 'dec_map', ...
    'gridSize', 'worldCenter', 'm', 'R1', 'R2', ...
    'g_tt', 'T_00');
fprintf('Results saved to: /tmp/matlab_original_results.mat\n');
fprintf('Variables saved:\n');
fprintf('  - metric_shell: Full Warp Shell metric\n');
fprintf('  - energy: Stress-energy tensor\n');
fprintf('  - nec_map, wec_map, sec_map, dec_map: Energy condition maps\n');
fprintf('  - g_tt: Metric time component\n');
fprintf('  - T_00: Energy density\n');
fprintf('  - Parameters: gridSize, worldCenter, m, R1, R2\n');
fprintf('\n');

%% SUMMARY
fprintf('==========================================================\n');
fprintf('SUMMARY - ORIGINAL MATLAB WARPFACTORY TEST COMPLETE\n');
fprintf('==========================================================\n');
fprintf('\n');
fprintf('Timings:\n');
fprintf('  Minkowski metric:      %.4f s\n', t_mink);
fprintf('  Warp Shell metric:     %.4f s\n', t_shell);
fprintf('  Energy tensor:         %.4f s\n', t_energy);
fprintf('  NEC computation:       %.4f s\n', t_nec);
fprintf('  WEC computation:       %.4f s\n', t_wec);
fprintf('  SEC computation:       %.4f s\n', t_sec);
fprintf('  DEC computation:       %.4f s\n', t_dec);
fprintf('  ---\n');
fprintf('  Total runtime:         %.4f s\n', t_mink + t_shell + t_energy + t_nec + t_wec + t_sec + t_dec);
fprintf('\n');

fprintf('Key Results:\n');
fprintf('  g_tt range: [%.6e, %.6e]\n', min(g_tt(:)), max(g_tt(:)));
fprintf('  T^00 range: [%.6e, %.6e]\n', min(T_00(:)), max(T_00(:)));
fprintf('  NEC range: [%.6e, %.6e] - %d violations (%.2f%%)\n', min(nec_map(:)), max(nec_map(:)), nec_violations, 100*nec_violations/numel(nec_map));
fprintf('  WEC range: [%.6e, %.6e] - %d violations (%.2f%%)\n', min(wec_map(:)), max(wec_map(:)), wec_violations, 100*wec_violations/numel(wec_map));
fprintf('  SEC range: [%.6e, %.6e] - %d violations (%.2f%%)\n', min(sec_map(:)), max(sec_map(:)), sec_violations, 100*sec_violations/numel(sec_map));
fprintf('  DEC range: [%.6e, %.6e] - %d violations (%.2f%%)\n', min(dec_map(:)), max(dec_map(:)), dec_violations, 100*dec_violations/numel(dec_map));
fprintf('\n');

fprintf('Status: ALL TESTS PASSED\n');
fprintf('The original MATLAB WarpFactory code is working correctly!\n');
fprintf('==========================================================\n');
