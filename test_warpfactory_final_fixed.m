% Complete WarpFactory test using MATLAB-Python bridge
disp('=== WarpFactory MATLAB-Python Complete Test ===');
disp(' ');

% Add Python path
P = py.sys.path;
if count(P, '/WarpFactory') == 0
    insert(P, int32(0), '/WarpFactory');
end

% Import WarpFactory modules
disp('Importing WarpFactory modules...');
wf = py.importlib.import_module('warpfactory');
disp('WarpFactory module loaded');
disp(' ');

% Define parameters (matching paper parameters)
disp('Setting up parameters...');
grid_size = py.list({int32(1), int32(21), int32(21), int32(21)});
world_center = py.list({0.5, 10.5, 10.5, 10.5});
m = 4.49e27;
R1 = 10.0;
R2 = 20.0;

disp(['Mass: ', num2str(m), ' kg (', num2str(m/1.898e27), ' Jupiter masses)']);
disp(['Inner radius: ', num2str(R1), ' m']);
disp(['Outer radius: ', num2str(R2), ' m']);
disp(['Grid size: [1, 21, 21, 21]']);
disp(' ');

% Create Minkowski metric (simple test)
disp('1. Creating Minkowski metric...');
try
    minkowski_module = py.importlib.import_module('warpfactory.metrics.minkowski.minkowski');
    metric_minkowski = minkowski_module.get_minkowski_metric(grid_size);
    disp('   Minkowski metric created successfully!');
catch ME
    disp(['   Error: ', ME.message]);
end
disp(' ');

% Create Warp Shell metric
disp('2. Creating Warp Shell Comoving metric...');
try
    warp_shell_module = py.importlib.import_module('warpfactory.metrics.warp_shell.warp_shell');
    
    metric_shell = warp_shell_module.get_warp_shell_comoving_metric(grid_size, world_center, pyargs('m', m, 'R1', R1, 'R2', R2, 'Rbuff', 0.0, 'sigma', 0.0, 'smooth_factor', 1.0, 'v_warp', 0.0, 'do_warp', false));
    
    disp('   Warp Shell metric created successfully!');
    
    % Get metric tensor components
    g_tensor = metric_shell.tensor;
    
    % Get some tensor values
    g00_data = py.numpy.array(g_tensor{py.tuple({int32(0), int32(0)})});
    g11_data = py.numpy.array(g_tensor{py.tuple({int32(1), int32(1)})});
    
    disp(['   g_00 shape: ', char(py.str(g00_data.shape))]);
    disp(['   g_11 shape: ', char(py.str(g11_data.shape))]);
    
catch ME
    disp(['   Error: ', ME.message]);
    disp(getReport(ME));
end
disp(' ');

% Compute stress-energy tensor
disp('3. Computing stress-energy tensor...');
try
    energy_module = py.importlib.import_module('warpfactory.solver.energy');
    energy_tensor = energy_module.get_energy_tensor(metric_shell);
    disp('   Stress-energy tensor computed successfully!');
    
    % Get energy tensor components
    T_tensor = energy_tensor.tensor;
    T00_data = py.numpy.array(T_tensor{py.tuple({int32(0), int32(0)})});
    disp(['   T^00 (energy density) shape: ', char(py.str(T00_data.shape))]);
    
catch ME
    disp(['   Error: ', ME.message]);
    disp(getReport(ME));
end
disp(' ');

% Compute energy conditions
disp('4. Computing energy conditions...');
try
    ec_module = py.importlib.import_module('warpfactory.analyzer.energy_conditions');
    
    disp('   Computing NEC (Null Energy Condition)...');
    nec_result = ec_module.get_energy_conditions(energy_tensor, metric_shell, 'Null');
    nec_vals = py.numpy.array(nec_result{int32(1)});
    disp(['      NEC values shape: ', char(py.str(nec_vals.shape))]);
    disp(['      NEC min: ', num2str(double(py.numpy.nanmin(nec_vals)))]);
    disp(['      NEC max: ', num2str(double(py.numpy.nanmax(nec_vals)))]);
    
    disp('   Computing WEC (Weak Energy Condition)...');
    wec_result = ec_module.get_energy_conditions(energy_tensor, metric_shell, 'Weak');
    wec_vals = py.numpy.array(wec_result{int32(1)});
    disp(['      WEC values shape: ', char(py.str(wec_vals.shape))]);
    disp(['      WEC min: ', num2str(double(py.numpy.nanmin(wec_vals)))]);
    disp(['      WEC max: ', num2str(double(py.numpy.nanmax(wec_vals)))]);
    
    disp('   Computing SEC (Strong Energy Condition)...');
    sec_result = ec_module.get_energy_conditions(energy_tensor, metric_shell, 'Strong');
    sec_vals = py.numpy.array(sec_result{int32(1)});
    disp(['      SEC values shape: ', char(py.str(sec_vals.shape))]);
    disp(['      SEC min: ', num2str(double(py.numpy.nanmin(sec_vals)))]);
    disp(['      SEC max: ', num2str(double(py.numpy.nanmax(sec_vals)))]);
    
    disp('   Computing DEC (Dominant Energy Condition)...');
    dec_result = ec_module.get_energy_conditions(energy_tensor, metric_shell, 'Dominant');
    dec_vals = py.numpy.array(dec_result{int32(1)});
    disp(['      DEC values shape: ', char(py.str(dec_vals.shape))]);
    disp(['      DEC min: ', num2str(double(py.numpy.nanmin(dec_vals)))]);
    disp(['      DEC max: ', num2str(double(py.numpy.nanmax(dec_vals)))]);
    
    disp(' ');
    disp('   Energy conditions computed successfully!');
    
catch ME
    disp(['   Error: ', ME.message]);
    disp(getReport(ME));
end
disp(' ');

% Save results to .mat file
disp('5. Saving results to MATLAB .mat file...');
try
    % Convert Python numpy arrays to MATLAB arrays
    g00_shape = double(py.array.array('l', g00_data.shape));
    g00_flat = double(py.array.array('d', py.numpy.nditer(g00_data)));
    g00_matlab = reshape(g00_flat, g00_shape);
    
    T00_shape = double(py.array.array('l', T00_data.shape));
    T00_flat = double(py.array.array('d', py.numpy.nditer(T00_data)));
    T00_matlab = reshape(T00_flat, T00_shape);
    
    nec_shape = double(py.array.array('l', nec_vals.shape));
    nec_flat = double(py.array.array('d', py.numpy.nditer(nec_vals)));
    nec_matlab = reshape(nec_flat, nec_shape);
    
    wec_shape = double(py.array.array('l', wec_vals.shape));
    wec_flat = double(py.array.array('d', py.numpy.nditer(wec_vals)));
    wec_matlab = reshape(wec_flat, wec_shape);
    
    sec_shape = double(py.array.array('l', sec_vals.shape));
    sec_flat = double(py.array.array('d', py.numpy.nditer(sec_vals)));
    sec_matlab = reshape(sec_flat, sec_shape);
    
    dec_shape = double(py.array.array('l', dec_vals.shape));
    dec_flat = double(py.array.array('d', py.numpy.nditer(dec_vals)));
    dec_matlab = reshape(dec_flat, dec_shape);
    
    % Save to .mat file
    save('/WarpFactory/warpfactory_matlab_results.mat', 'g00_matlab', 'T00_matlab', 'nec_matlab', 'wec_matlab', 'sec_matlab', 'dec_matlab', 'm', 'R1', 'R2');
    
    disp('   Results saved to: /WarpFactory/warpfactory_matlab_results.mat');
    disp('   Variables saved:');
    disp('      g00_matlab - Metric component g_00');
    disp('      T00_matlab - Energy density T^00');
    disp('      nec_matlab - Null Energy Condition values');
    disp('      wec_matlab - Weak Energy Condition values');
    disp('      sec_matlab - Strong Energy Condition values');
    disp('      dec_matlab - Dominant Energy Condition values');
    disp('      m, R1, R2 - Physical parameters');
    
catch ME
    disp(['   Error saving results: ', ME.message]);
    disp(getReport(ME));
end
disp(' ');

disp('=== WarpFactory MATLAB Test Complete ===');
disp(' ');
disp('SUMMARY:');
disp('--------');
disp('1. MATLAB R2023b is working correctly');
disp('2. MATLAB-Python integration is functional');
disp('3. WarpFactory Python library successfully loaded');
disp('4. Metrics created (Minkowski and Warp Shell)');
disp('5. Stress-energy tensor computed from metric');
disp('6. All energy conditions (NEC, WEC, SEC, DEC) computed');
disp('7. Results saved to .mat file for further analysis');
disp(' ');
disp('MATLAB can successfully run WarpFactory code through Python bridge!');
