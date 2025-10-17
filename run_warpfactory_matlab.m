% Run WarpFactory using MATLAB-Python bridge
% This script calls the Python WarpFactory implementation from MATLAB

disp('=== WarpFactory MATLAB-Python Test ===');
disp(' ');

% Add Python path
py.sys.path.append('/WarpFactory');

% Import WarpFactory modules
disp('Importing WarpFactory modules...');
wf = py.importlib.import_module('warpfactory');

% Define parameters (matching paper parameters)
disp('Setting up parameters...');
grid_size = py.list({int32(1), int32(21), int32(21), int32(21)});
world_center = py.list({0.5, 10.5, 10.5, 10.5});
m = 4.49e27;  % Total mass (kg)
R1 = 10.0;    % Inner radius (m)
R2 = 20.0;    % Outer radius (m)

disp(['Grid size: ', char(py.str(grid_size))]);
disp(['World center: ', char(py.str(world_center))]);
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

    % Get tensor component
    g00 = metric_minkowski.tensor;
    disp(['Metric tensor type: ', char(py.type(g00).__name__)]);
catch ME
    disp(['Error creating Minkowski metric: ', ME.message]);
end
disp(' ');

% Create Warp Shell metric
disp('Creating Warp Shell Comoving metric...');
try
    warp_shell_module = py.importlib.import_module('warpfactory.metrics.warp_shell.warp_shell');

    % Call with paper parameters
    metric_shell = warp_shell_module.get_warp_shell_comoving_metric(...
        grid_size, ...
        world_center, ...
        pyargs(...
            'm', m, ...
            'R1', R1, ...
            'R2', R2, ...
            'Rbuff', 0.0, ...
            'sigma', 0.0, ...
            'smooth_factor', 1.0, ...
            'v_warp', 0.0, ...
            'do_warp', false ...
        ));

    disp('Warp Shell metric created successfully!');
    disp(['Metric type: ', char(py.type(metric_shell).__name__)]);

    % Access tensor components
    tensor = metric_shell.tensor;
    disp(['Tensor type: ', char(py.type(tensor).__name__)]);

catch ME
    disp(['Error creating Warp Shell metric: ', ME.message]);
    disp('Stack trace:');
    disp(getReport(ME));
end
disp(' ');

% Compute stress-energy tensor
disp('Computing stress-energy tensor...');
try
    solver_module = py.importlib.import_module('warpfactory.solver.energy_tensor');
    energy_tensor = solver_module.get_energy_tensor(metric_shell);
    disp('Stress-energy tensor computed successfully!');
    disp(['Energy tensor type: ', char(py.type(energy_tensor).__name__)]);
catch ME
    disp(['Error computing stress-energy tensor: ', ME.message]);
    disp('Stack trace:');
    disp(getReport(ME));
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
    disp('Stack trace:');
    disp(getReport(ME));
end
disp(' ');

disp('=== WarpFactory MATLAB Test Complete ===');
