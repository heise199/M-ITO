# Log Format Documentation

## Overview
Both MATLAB and Python versions now output unified log format for easy comparison.

## Log Tags

### Initialization Phase

**[INPUT]** - Input parameters
```
[INPUT] L=10.00, W=5.00, Order=[1,1], Num=[6,4]
[INPUT] BoundCon=1, Vmax=0.200, penal=3.0, rmin=2.0
```

**[NURBS]** - NURBS geometry information
```
[NURBS] number=[12,7], order=[3,3]
[NURBS] knots[0] length=15, range=[0.000000, 1.000000]
[NURBS] knots[1] length=10, range=[0.000000, 1.000000]
```

**[IGA]** - IGA setup (dimensions, DOFs)
```
[IGA] Dim=2, Dofs.Num=168
```

**[LOAD]** - Load vector information
```
[LOAD] F nonzero=9, sum(abs(F))=1.000000e+00, max(abs(F))=4.549451e-01
```

**[BASIS]** - Basis function matrices
```
[BASIS] N size=[450,9], id size=[450,9]
[BASIS] R size=[450,84], nnz(R)=4050
[BASIS] dRu size=[450,9], dRv size=[450,9]
```

### Iteration Phase

**========== ITERATION X ==========** - Start of each iteration

**[STIFF]** - Element stiffness matrix information
```
[STIFF] KE{1} size=[18,18], norm(KE{1})=1.234567e+00
[STIFF] dv_dg: min=1.234567e-03, max=9.876543e-02, sum=5.432100e+01
```

**[ASS]** - Assembled global stiffness matrix
```
[ASS] K size=[168,168], nnz(K)=12345, norm(K)=9.876543e+02
```

**[SOLVE]** - Displacement solution
```
[SOLVE] U: min=-1.234567e-02, max=5.432100e-01, norm=3.456789e-01
```

**[OBJ]** - Objective function value
```
[OBJ] J=1.234567e+02, mean(X.GauPts)=0.500000
```

**[SENS]** - Sensitivity information
```
[SENS] dJ_dg: min=-5.432100e+01, max=1.234567e+02, sum=-9.876543e+03
[SENS] dJ_dp: min=-8.765432e+00, max=3.456789e+01, sum=-1.234567e+03
[SENS] dv_dp: min=1.234567e-03, max=9.876543e-01, sum=5.432100e+02
```

**[OC]** - Optimality criteria update
```
[OC] X.CtrPts_new: min=0.100000, max=1.000000, mean=0.450000
```

**[CHANGE]** - Design variable change
```
[CHANGE] change=2.000000e-01
```

## Comparison Points

### Critical Values to Compare

1. **Initial Setup**
   - NURBS.number and NURBS.order should be identical
   - F nonzero count and sum should match
   - R matrix size and nnz should be the same

2. **First Iteration**
   - norm(KE{1}) - should match within tolerance (~1e-10)
   - sum(dv_dg) - should be identical
   - norm(K) - should match
   - norm(U) - critical indicator
   - J (objective) - should be very close
   - Sensitivity sums - should match

3. **Subsequent Iterations**
   - J should converge to same value
   - change should be similar

## How to Compare

### Step 1: Run both versions
```bash
# MATLAB
cd M_ITO
Test_Log_Comparison

# Python
cd M_ITO_python
python test_log_comparison.py
```

### Step 2: Extract key values
Copy the output to text files:
- `matlab_log.txt`
- `python_log.txt`

### Step 3: Compare systematically
Look for first difference:
1. Check [INPUT] - parameters should be identical
2. Check [NURBS] - geometry should match
3. Check [LOAD] - if F is different, problem is in boun_cond
4. Check [BASIS] - if R differs, problem is in nrbbasisfun
5. Check iteration 1 [STIFF] - if different, problem is in stiff_ele2d
6. And so on...

## Expected Tolerances

- **Integer values** (sizes, counts): Should be **exactly** the same
- **Floating point values**: 
  - Small values (< 1e-6): Relative error < 1e-8
  - Medium values (1e-6 to 1e6): Relative error < 1e-10
  - Large values (> 1e6): Relative error < 1e-8

## Common Issues

### Issue 1: F (load) is zero in Python but not in MATLAB
**Cause**: Zero-weight control points at boundary
**Location**: boun_cond.py line 34, 44, 60, 68
**Fix**: Load points shifted from 1.0 to 0.99

### Issue 2: Different matrix sizes
**Cause**: Array indexing or reshape order
**Location**: Check pre_iga.py, especially Ele.Seque and Ele.CtrPtsCon

### Issue 3: U (displacement) all zeros
**Cause**: K matrix singular or wrong boundary conditions
**Check**: [ASS] and [SOLVE] logs

### Issue 4: J increases instead of decreases
**Cause**: Sensitivity calculation wrong
**Check**: [SENS] logs, especially signs of dJ_dg

## Debug Strategy

If outputs differ:

1. **Find the first difference** - use binary search through logs
2. **Isolate the module** - which [TAG] first shows difference?
3. **Add detailed logs** - modify that specific module
4. **Compare intermediate values** - add prints inside loops if needed
5. **Check array order** - print first few elements, check order='F'

## Example Comparison

```
MATLAB:                          Python:
[INPUT] L=10.00, W=5.00         [INPUT] L=10.00, W=5.00     ✓ Match
[NURBS] number=[12,7]           [NURBS] number=[12,7]       ✓ Match
[LOAD] sum(abs(F))=1.000000e+00 [LOAD] sum(abs(F))=1.000000e+00  ✓ Match
[OBJ] J=123.4567                [OBJ] J=0.0000              ✗ DIFFER!
                                                             → Check stiff_ele2d
```

