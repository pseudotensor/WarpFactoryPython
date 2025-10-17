% Run WarpFactory using MATLAB-Python bridge
disp('=== WarpFactory MATLAB-Python Test ===');
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

disp(['Mass: ', num2str(m), ' kg']);
disp(['Inner radius: ', num2str(R1), ' m']);
disp(['Outer radius: ', num2str(R2), ' m']);
disp(' ');

% Create Minkowski metric (simple test)
disp('Creating Minkowski metric...');
try
    minkowski_module = py.importlib.import_module('warpfactory.metrics.minkowski.minkowski');
    metric_minkowski = minkowski_module.get_minkowski_metric(grid_size);
    disp('Minkowski metric created successfully!');
catch ME
    disp(['Error creating Minkowski metric: ', ME.message]);
end
disp(' ');

% Create Warp Shell metric
disp('Creating Warp Shell Comoving metric...');
try
    warp_shell_module = py.importlib.import_module('warpfactory.metrics.warp_shell.warp_shell');
    
    % Call with paper parameters
    metric_shell = warp_shell_module.get_warp_shell_comoving_metric(grid_size, world_center, pyargs('m', m, 'R1', R1, 'R2', R2, 'Rbuff', 0.0, 'sigma', 0.0, 'smooth_factor', 1.0, 'v_warp', 0.0, 'do_warp', false));
    
    disp('Warp Shell metric created successfully!');
catch ME
    disp(['Error creating Warp Shell metric: ', ME.message]);
end
disp(' ');

% Compute stress-energy tensor
disp('Computing stress-energy tensor...');
try
    solver_module = py.importlib.import_module('warpfactory.solver.energy_tensor');
    energy_tensor = solver_module.get_energy_tensor(metric_shell);
    disp('Stress-energy tensor computed successfully!');
catch ME
    disp(['Error computing stress-energy tensor: ', ME.message]);
end
disp(' ');

% Compute energy conditions
disp('Computing energy conditions...');
try
    analyzer_module = py.importlib.import_module('warpfactory.analyzer.energy_conditions');
    
    % Compute NEC (Null Energy Condition)
    nec = analyzer_module.get_energy_conditions(energy_tensor, metric_shell, 'Null');
    disp('NEC computed successfully!');
    
    % Compute WEC (Weak Energy Condition)
    wec = analyzer_module.get_energy_conditions(energy_tensor, metric_shell, 'Weak');
    disp('WEC computed successfully!');
    
    % Compute SEC (Strong Energy Condition)
    sec = analyzer_module.get_energy_conditions(energy_tensor, metric_shell, 'Strong');
    disp('SEC computed successfully!');
    
    % Compute DEC (Dominant Energy Condition)
    dec = analyzer_module.get_energy_conditions(energy_tensor, metric_shell, 'Dominant');
    disp('DEC computed successfully!');
    
catch ME
    disp(['Error computing energy conditions: ', ME.message]);
end
disp(' ');

disp('=== WarpFactory MATLAB Test Complete ===');
