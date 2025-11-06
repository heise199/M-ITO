#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to match MATLAB Michell-type structure case
IgaTop2D(10, 4, [1 1], [101 41], 3, 0.2, 3, 2)
"""

import sys
import numpy as np
from iga_top2d import iga_top2d

def main():
    print("\n" + "="*80)
    print("Python Michell-type Structure Test - Matching MATLAB")
    print("="*80)
    print("\nTest parameters:")
    print("  Geometry: L=10, W=4")
    print("  NURBS: Order=[1, 1], Num=[101, 41]")
    print("  Boundary condition: 3 (Michell-type)")
    print("  Optimization: Vmax=0.2, penal=3, rmin=2")
    print("\n")
    
    # Run the optimization - matching MATLAB parameters exactly
    iga_top2d(
        L=10.0,
        W=4.0,
        Order=np.array([1, 1]),
        Num=np.array([101, 41]),
        BoundCon=3,
        Vmax=0.2,
        penal=3.0,
        rmin=2.0,
        MaxIter=2  # Just run 2 iterations to check initial results
    )
    
    print("\n" + "="*80)
    print("Python test completed. Compare with MATLAB output.")
    print("="*80)

if __name__ == '__main__':
    main()

