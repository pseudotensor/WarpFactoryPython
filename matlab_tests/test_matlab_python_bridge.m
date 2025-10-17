% Test MATLAB-Python bridge for WarpFactory
disp('=== Testing MATLAB-Python Integration ===');

% Check Python configuration
disp('Python Configuration:');
pyversion

% Test if we can import Python
try
    disp('Testing Python import...');
    py.print('Hello from Python!');
    disp('Python import successful!');
catch ME
    disp(['Error: ', ME.message]);
    error('Cannot access Python from MATLAB');
end

disp('=== MATLAB-Python Integration Test Complete ===');
