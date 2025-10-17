% Quick verification of MATLAB results
disp('=== MATLAB Results Verification ===');
disp(' ');

% Load results
load('/WarpFactory/warpfactory_matlab_results.mat');

% Display parameters
disp('Physical Parameters:');
disp(['  Mass: ', num2str(m), ' kg (', num2str(m/1.898e27), ' Jupiter masses)']);
disp(['  Inner radius: ', num2str(R1), ' m']);
disp(['  Outer radius: ', num2str(R2), ' m']);
disp(' ');

% Display array sizes
disp('Data Dimensions:');
disp(['  g00_matlab: ', num2str(size(g00_matlab))]);
disp(['  T00_matlab: ', num2str(size(T00_matlab))]);
disp(['  nec_matlab: ', num2str(size(nec_matlab))]);
disp(' ');

% Statistical analysis
disp('Metric Component g_00:');
disp(['  Min: ', num2str(min(g00_matlab(:)))]);
disp(['  Max: ', num2str(max(g00_matlab(:)))]);
disp(['  Mean: ', num2str(mean(g00_matlab(:)))]);
disp(' ');

disp('Energy Density T^00:');
disp(['  Min: ', num2str(min(T00_matlab(:)))]);
disp(['  Max: ', num2str(max(T00_matlab(:)))]);
disp(['  Mean: ', num2str(mean(T00_matlab(:), 'omitnan'))]);
disp(['  Non-zero points: ', num2str(sum(abs(T00_matlab(:)) > 1e30))]);
disp(' ');

% Energy condition violations
disp('Energy Condition Analysis:');
total_points = numel(nec_matlab);
nec_violations = sum(nec_matlab(:) < 0);
wec_violations = sum(wec_matlab(:) < 0);
sec_violations = sum(sec_matlab(:) < 0);
dec_violations = sum(dec_matlab(:) < 0);

disp(['  Total grid points: ', num2str(total_points)]);
disp(['  NEC violations: ', num2str(nec_violations), ' (', num2str(100*nec_violations/total_points, '%.2f'), '%)']);
disp(['  WEC violations: ', num2str(wec_violations), ' (', num2str(100*wec_violations/total_points, '%.2f'), '%)']);
disp(['  SEC violations: ', num2str(sec_violations), ' (', num2str(100*sec_violations/total_points, '%.2f'), '%)']);
disp(['  DEC violations: ', num2str(dec_violations), ' (', num2str(100*dec_violations/total_points, '%.2f'), '%)']);
disp(' ');

disp('=== Verification Complete ===');
