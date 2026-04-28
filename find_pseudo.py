#!/usr/bin/env python3
"""
Find all N-segment Rubik's Snake pseudo loops.
Usage: python find_pseudo.py N   (N must be odd)

A pseudo loop is a non-self-intersecting open snake of N segments whose
first and last square faces are opposite sides of a unit cube:
  - T_last y-column ≈ (-1, 0, 0)
  - T_last[:3,:3] @ [0.5,0,0.5] + T_last[:3,3] ≈ (-1, 0.5, 0.5)

Equivalences: reversal and swap13 (1↔3, 0 and 2 fixed).
"""

import math, sys, time
import numpy as np

PI = math.pi

def _T(tx, ty, tz):
    m = np.eye(4); m[0,3]=tx; m[1,3]=ty; m[2,3]=tz; return m

def _build_joint(parity, d):
    theta = (2 - d) * PI / 2; c, s = math.cos(theta), math.sin(theta)
    Rx    = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], float)
    Ry    = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], float)
    Rx_pi = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    Ry_pi = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]], float)
    if parity == 0: return _T(0.5,0,0.5) @ Ry @ Rx_pi @ _T(-0.5,0,-0.5)
    else:           return _T(0,0.5,0.5) @ Rx @ Ry_pi @ _T(0,-0.5,-0.5)

JOINTS = [[_build_joint(p, d) for d in range(4)] for p in range(2)]

_SWAP13 = (0, 3, 2, 1)

_SAMPLES_LOCAL = np.array([
    [0.20, 0.20, 0.50], [0.35, 0.10, 0.25], [0.10, 0.35, 0.25],
    [0.35, 0.10, 0.75], [0.10, 0.35, 0.75],
], float)
_EPS = 0.08

def _wsamples(T):
    return (_SAMPLES_LOCAL @ T[:3,:3].T) + T[:3,3]

def _intersects(ws_i, inv_j):
    p = (ws_i @ inv_j[:3,:3].T) + inv_j[:3,3]
    return bool(np.any(
        (p[:,0] > _EPS) & (p[:,1] > _EPS) & (p[:,2] > _EPS) &
        (p[:,2] < 1 - _EPS) & (p[:,0] + p[:,1] < 1 - _EPS)
    ))

TARGET_YCOL   = np.array([-1., 0., 0.])
TARGET_CENTER = np.array([-1., 0.5, 0.5])
LOCAL_CENTER  = np.array([0.5, 0., 0.5])

def is_pseudo_loop(T):
    if not np.allclose(T[:3, 1], TARGET_YCOL, atol=1e-5):
        return False
    return np.allclose(T[:3, :3] @ LOCAL_CENTER + T[:3, 3], TARGET_CENTER, atol=1e-5)

def has_open_pair(ws_stack, n):
    """At least one opposite pair of the pseudo loop cube's 4 side faces must be clear.

    Cube: x∈[-1,0], y∈[0,1], z∈[0,1]. Side faces: y=0, y=1, z=0, z=1.
    A face is blocked if any segment has a sample point outside that face's plane
    and within the face's x and lateral extent.
    """
    by0 = by1 = bz0 = bz1 = False
    for i in range(n):
        for x, y, z in ws_stack[i]:
            if -1 <= x <= 0:
                if 0 <= z <= 1:
                    if y < 0: by0 = True
                    if y > 1: by1 = True
                if 0 <= y <= 1:
                    if z < 0: bz0 = True
                    if z > 1: bz1 = True
    return (not by0 and not by1) or (not bz0 and not bz1)

def get_variants(digits):
    d  = tuple(digits)
    r  = d[::-1]
    s  = tuple(_SWAP13[x] for x in d)
    rs = tuple(_SWAP13[x] for x in r)
    return [d, r, s, rs]

def canonical(digits):
    return min(get_variants(digits))

def find_pseudo(N):
    assert N % 2 == 1, "N must be odd"
    n_digits = N - 1

    found = {}
    T_stack   = [None] * (n_digits + 1)
    ws_stack  = [None] * (n_digits + 1)
    inv_stack = [None] * (n_digits + 1)
    T_stack[0]   = np.eye(4)
    ws_stack[0]  = _wsamples(np.eye(4))
    inv_stack[0] = np.eye(4)
    digits = [0] * n_digits

    def dfs(depth):
        if depth == n_digits:
            if is_pseudo_loop(T_stack[depth]) and has_open_pair(ws_stack, depth + 1):
                dig = tuple(digits)
                canon = canonical(dig)
                if canon not in found:
                    found[canon] = dig
            return

        parity = depth % 2
        for d in range(4):
            T_new = T_stack[depth] @ JOINTS[parity][d]
            nd    = depth + 1
            ws    = _wsamples(T_new)
            iv    = np.linalg.inv(T_new)

            bad = False
            for j in range(nd - 2):
                if _intersects(ws, inv_stack[j]) or _intersects(ws_stack[j], iv):
                    bad = True; break
            if bad:
                continue

            T_stack[nd]  = T_new
            ws_stack[nd] = ws
            inv_stack[nd]= iv
            digits[depth]= d
            dfs(nd)

    t0 = time.time()
    dfs(0)
    return found, time.time() - t0

def main():
    N = int(sys.argv[1])
    found, elapsed = find_pseudo(N)
    results = sorted(found.keys())
    print(f"# {N}-segment pseudo loops: {len(found)} found in {elapsed:.1f}s")
    for i, canon in enumerate(results, 1):
        print(f"loop{N}_{i} {''.join(map(str, canon))}")

if __name__ == '__main__':
    main()
