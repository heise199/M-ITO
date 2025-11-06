% Test script for log comparison - using small-scale case
% Uses same parameters as Python version

fprintf('\nRunning small-scale test for log comparison...\n');
fprintf('Parameters: L=10, W=5, Order=[0,0], Num=[6,4], BoundCon=1, Vmax=0.2, penal=3, rmin=2\n\n');

% Call main function
IgaTop2D(10, 5, [0, 0], [6, 4], 1, 0.2, 3, 2);

fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('MATLAB test completed, please compare output with Python\n');
