'''
Accuracy tests for the double-double (WITH_DOUBLE_DOUBLE) build of the
extended-precision Rys quadrature.

    python test_double_double.py --dd build-dd/libcint.so \
        [--quad build-quad/libcint.so] [--ld build-ld/libcint.so]

Configure the libraries with -DKEEP_GOING=1: some existing short-range
cases report failure and would otherwise call exit().

1. Boys functions F_m(t) of the double-double build (fmt_dderfc_like)
   against mpmath.
2. CINTrys_roots for nroots >= 13 (which uses CINTqrys_jacobi and
   CINTqrys_laguerre) against mpmath.  The double-double build must be
   accurate to ~1e-12 like the quadmath build.
3. CINTsr_rys_roots for nroots >= 7 against mpmath.  These cases are
   limited by the working precision even with quadmath; double-double has
   7 fewer bits than binary128, so ~2 digits are lost relative to the
   quadmath build.  Errors are reported, not asserted.
'''
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../scripts')))

import argparse
import ctypes
import numpy as np
import mpmath
import rys_roots

mpmath.mp.dps = 250
ERFC_bound = 200

def rys_call(lib, nroots, x, low):
    r = np.zeros(nroots)
    w = np.zeros(nroots)
    if low == 0:
        lib.CINTrys_roots(ctypes.c_int(nroots), ctypes.c_double(x),
                          r.ctypes.data_as(ctypes.c_void_p),
                          w.ctypes.data_as(ctypes.c_void_p))
    else:
        lib.CINTsr_rys_roots(ctypes.c_int(nroots), ctypes.c_double(x), ctypes.c_double(low),
                             r.ctypes.data_as(ctypes.c_void_p),
                             w.ctypes.data_as(ctypes.c_void_p))
    return r, w

def dd_fmt(lib, t, low, m):
    f = np.zeros((m + 1, 2))
    lib.fmt_dderfc_like(f.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(t),
                        ctypes.c_double(low), ctypes.c_int(m))
    return [mpmath.mpf(hi) + mpmath.mpf(lo) for hi, lo in f]

def rel_err(a, ref):
    scale = abs(ref).max()
    if scale == 0:
        return abs(a).max()
    return abs(a - ref).max() / scale

def check_fmt(lib):
    '''Boys function F_m(t) (and its erfc-attenuated variant) in double-double'''
    m = 24
    worst = 0
    for t in [0., 1e-8, 0.03, 0.5, 1.7, 3.3, 7.9, 15., 30., 61., 120., 250.]:
        for low in [0., 0.1, 0.3, 0.6, 0.9]:
            if t * low**2 > ERFC_bound:
                continue  # set to 0 by design
            f = dd_fmt(lib, t, low, m)
            if low == 0:
                ref = rys_roots.fmt(mpmath.mpf(t), m)
            else:
                ref = rys_roots.fmt_erfc(mpmath.mpf(t), m, mpmath.mpf(low))
            for i in range(m + 1):
                err = abs(f[i] / ref[i] - 1)
                if err > worst:
                    worst = err
                    where = (t, low, i)
    # The recursions amplify rounding errors by up to ~1e4 near low ~ 1
    # (quadmath: 1.2e-30 at t=7.9, low=0.9), so ~1e-28 is expected here.
    ok = worst < 1e-27
    print(f'fmt_dderfc_like  max rel err {mpmath.nstr(worst, 3)} at (t, low, m) = {where}'
          f'  {"OK" if ok else "FAILED"}')
    return ok

def reference(nroots, x, low):
    r, w = rys_roots.rys_roots_weights(nroots, mpmath.mpf(x),
                                       None if low == 0 else mpmath.mpf(low))
    return np.array(r, dtype=float), np.array(w, dtype=float)

def errors(libs, pts):
    errs = {name: [] for name in libs}
    for nroots, x, low in pts:
        try:
            r_ref, w_ref = reference(nroots, x, low)
        except RuntimeError:  # reference loses precision
            for name in libs:
                errs[name].append(np.nan)
            continue
        for name, lib in libs.items():
            r, w = rys_call(lib, nroots, x, low)
            errs[name].append(max(rel_err(r, r_ref), rel_err(w, w_ref)))
    return {name: np.array(e) for name, e in errs.items()}

def summary(title, errs):
    for name, e in errs.items():
        e = e[~np.isnan(e)]
        print(f'{title:16s} {name:4s} max {e.max():.2e}  median {np.median(e):.2e}'
              f'  #(err > 1e-10) {int((e > 1e-10).sum())}/{len(e)}')

def check_rys_roots(libs):
    xs = list(2.**np.arange(-5, 9)) + [1e-6, 1e-3, 700., 2000., 1e4, 1e5]
    pts = [(n, x, 0.) for n in (13, 14, 16, 20, 24) for x in xs]
    errs = errors(libs, pts)
    summary('CINTrys_roots', errs)
    worst = np.nanmax(errs['dd'])
    ok = worst < 1e-12
    print(f'CINTrys_roots    dd max err {worst:.2e}  {"OK" if ok else "FAILED"}')
    return ok

def report_sr_rys_roots(libs):
    xs = 2.**np.arange(-5, 9)
    pts = [(n, x, low) for n in (8, 10, 12, 14, 16, 20, 24)
           for low in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97) for x in xs if x * low**2 <= 40]
    errs = errors(libs, pts)
    summary('CINTsr_rys_roots', errs)
    if 'quad' in errs:
        mask = (errs['quad'] > 1e-11) & (errs['quad'] < 1e-3)
        ratio = np.log10(errs['dd'][mask] / errs['quad'][mask])
        print(f'CINTsr_rys_roots log10(dd err / quad err) where quad is precision limited: '
              f'median {np.median(ratio):.2f} (7 bits = {np.log10(128):.2f})')
    return not np.isnan(errs['dd']).all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dd', required=True)
    parser.add_argument('--quad')
    parser.add_argument('--ld')
    args = parser.parse_args()

    dd = ctypes.CDLL(os.path.abspath(args.dd))
    libs = {'dd': dd}
    if args.quad:
        libs['quad'] = ctypes.CDLL(os.path.abspath(args.quad))
    if args.ld:
        libs['ld'] = ctypes.CDLL(os.path.abspath(args.ld))

    ok = check_fmt(dd)
    ok &= check_rys_roots(libs)
    ok &= report_sr_rys_roots(libs)
    print('OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)
