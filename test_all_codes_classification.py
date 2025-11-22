#!/usr/bin/env python3
"""
Test that PLU-based error classification correctly identifies stabilizer errors
for all surface codes embedded in the HTML file.
"""

from src.qec.code_constructions.surface_code import SurfaceCode
from src.qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from ldpc.mod2 import PluDecomposition
import numpy as np


def test_stabilizer_classification(code, code_name):
    """
    Test that each row of S can be correctly classified as a stabilizer error
    using the PLU decomposition of S^T.
    """
    print(f"\n{'='*70}")
    print(f"Testing: {code_name}")
    print(f"{'='*70}")
    
    # Get stabilizer matrix
    S = code.stabilizer_matrix.toarray()
    S_T = S.T
    
    n_stabilizers = S.shape[0]
    n_error_positions = S.shape[1]
    n_qubits = code.physical_qubit_count
    
    print(f"  Code parameters: n={n_qubits}, k={code.logical_qubit_count}")
    print(f"  Stabilizer matrix S: {S.shape}")
    print(f"  S^T (for PLU): {S_T.shape}")
    
    # Compute PLU of S^T
    plu = PluDecomposition(S_T)
    print(f"  PLU rank: {plu.rank}/{n_stabilizers}")
    
    if plu.rank != n_stabilizers:
        print(f"  ⚠ WARNING: PLU rank {plu.rank} < {n_stabilizers} stabilizers")
    
    # Test each stabilizer (row of S)
    failures = []
    successes = 0
    
    for i in range(n_stabilizers):
        error_vector = S[i, :]
        
        # Check syndrome is zero (this might not be true - see notes below)
        syndrome = (S @ error_vector) % 2
        syndrome_is_zero = np.all(syndrome == 0)
        
        # Try to solve S^T @ coeffs = error_vector using PLU
        # We'll manually implement the forward-backward substitution
        
        # Apply permutation: Py = P @ y
        Py = (plu.P.toarray() @ error_vector) % 2
        
        # Forward substitution: Lb = Py
        L = plu.L.toarray()
        b = np.zeros(plu.rank, dtype=int)
        for k in range(plu.rank):
            s = 0
            for j in range(k):
                s ^= (L[k, j] * b[j])
            b[k] = Py[k] ^ s
        
        # Check if y is in image
        in_image = True
        for k in range(plu.rank, len(Py)):
            s = 0
            for j in range(plu.rank):
                s ^= (L[k, j] * b[j])
            if (Py[k] ^ s) != 0:
                in_image = False
                break
        
        if not in_image:
            failures.append({
                'stabilizer_idx': i,
                'reason': 'Not in image of S^T (PLU solver returns null)',
                'syndrome_zero': syndrome_is_zero
            })
            continue
        
        # Backward substitution: Ux = b
        U = plu.U.toarray()
        pivot_cols = plu.pivots
        coeffs = np.zeros(n_stabilizers, dtype=int)
        for k in range(plu.rank - 1, -1, -1):
            s = 0
            for j in range(pivot_cols[k] + 1, n_stabilizers):
                s ^= (U[k, j] * coeffs[j])
            coeffs[pivot_cols[k]] = b[k] ^ s
        
        # Verify reconstruction: S^T @ coeffs should equal error_vector
        reconstructed = (S_T @ coeffs) % 2
        matches = np.array_equal(reconstructed, error_vector)
        
        if not matches:
            failures.append({
                'stabilizer_idx': i,
                'reason': 'Reconstruction failed (S^T @ coeffs ≠ error)',
                'syndrome_zero': syndrome_is_zero,
                'coeffs': coeffs,
                'expected': error_vector,
                'got': reconstructed
            })
        else:
            successes += 1
            # Verify that coeffs[i] = 1 (we should need just the i-th stabilizer)
            if coeffs[i] != 1 or np.sum(coeffs) != 1:
                print(f"  ⚠ Stabilizer {i}: reconstructs correctly but uses coeffs={np.where(coeffs)[0]}")
    
    # Report results
    print(f"\n  Results:")
    print(f"    ✓ Successes: {successes}/{n_stabilizers}")
    if failures:
        print(f"    ✗ Failures: {len(failures)}/{n_stabilizers}")
        for fail in failures[:3]:  # Show first 3 failures
            print(f"      - Stabilizer {fail['stabilizer_idx']}: {fail['reason']}")
            print(f"        Syndrome zero: {fail['syndrome_zero']}")
        if len(failures) > 3:
            print(f"      ... and {len(failures) - 3} more")
    else:
        print(f"    ✓ All stabilizers correctly classified!")
    
    return len(failures) == 0


def main():
    print("="*70)
    print("TESTING: PLU-based Error Classification")
    print("Testing all surface codes that will be embedded in HTML")
    print("="*70)
    
    all_passed = True
    
    # Test Rotated Surface Codes (d=3,5,7,9,11)
    print("\n" + "="*70)
    print("ROTATED SURFACE CODES")
    print("="*70)
    for d in [3, 5, 7, 9, 11]:
        code = RotatedSurfaceCode(d)
        passed = test_stabilizer_classification(code, f"Rotated Surface Code d={d}")
        all_passed = all_passed and passed
    
    # Test Surface Codes (d=3-11)
    print("\n" + "="*70)
    print("SURFACE CODES")
    print("="*70)
    for d in range(3, 12):
        code = SurfaceCode(d)
        passed = test_stabilizer_classification(code, f"Surface Code d={d}")
        all_passed = all_passed and passed
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("All stabilizer errors correctly classified for all codes.")
    else:
        print("❌ SOME TESTS FAILED")
        print("See details above for which codes failed.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
