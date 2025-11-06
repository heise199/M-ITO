"""
Test script for log comparison - using small-scale case
"""

from iga_top2d import iga_top2d

# Use small-scale parameters for quick comparison
L = 10
W = 5  
Order = [0, 0]  # No degree elevation, keep degree 2
Num = [6, 4]    # Very small mesh
BoundCon = 1
Vmax = 0.2
penal = 3
rmin = 2

print('Running small-scale test for log comparison...')
print('Please run simultaneously in MATLAB:')
print(f'IgaTop2D({L}, {W}, [{Order[0]} {Order[1]}], [{Num[0]} {Num[1]}], {BoundCon}, {Vmax}, {penal}, {rmin})')
print('\n' + '='*80 + '\n')

X, Data, Iter_Ch = iga_top2d(L, W, Order, Num, BoundCon, Vmax, penal, rmin, case_number=999)

print('\n' + '='*80)
print('Python test completed, please compare output with MATLAB')
