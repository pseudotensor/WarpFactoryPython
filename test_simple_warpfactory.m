% Simple WarpFactory test using MATLAB-Python bridge
disp('=== Simple WarpFactory Test ===');

% Add Python path
py.sys.path.append('/WarpFactory');

% Import WarpFactory
disp('Importing WarpFactory...');
wf = py.importlib.import_module('warpfactory');
disp(['WarpFactory module loaded: ', char(py.type(wf).__name__)]);

% Import Minkowski metric module
disp('Importing Minkowski module...');
minkowski = py.importlib.import_module('warpfactory.metrics.minkowski.minkowski');

% Create simple grid
grid_size = py.list({int32(5), int32(10), int32(10), int32(10)});
disp(['Grid size: ', char(py.str(grid_size))]);

% Create Minkowski metric
disp('Creating Minkowski metric...');
metric = minkowski.get_minkowski_metric(grid_size);
disp('Minkowski metric created!');
disp(['Metric type: ', char(py.type(metric).__name__)]);

disp('=== Test Complete ===');
