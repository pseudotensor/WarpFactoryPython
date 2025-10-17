% Test Python-MATLAB bridge
disp('Testing Python import...');
py.sys.path.append('/WarpFactory');
wf = py.importlib.import_module('warpfactory');
disp('WarpFactory imported successfully!');
