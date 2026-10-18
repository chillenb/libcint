'''
Compare dd_real.h (via the test_dd_real driver) against mpmath.

    gcc -O2 -ffp-contract=off -Isrc testsuite/test_dd_real.c -o test_dd_real -lm
    ./test_dd_real | python testsuite/test_dd_real.py
'''
import sys
import mpmath as mp

mp.mp.dps = 50

def ref(name, x):
    x = mp.mpf(x)
    if name == 'expm':
        return mp.exp(-x)
    if name == 'erf':
        return mp.erf(x)
    if name == 'erfc':
        return mp.erfc(x)
    if name == 'sqrt':
        return mp.sqrt(x)
    if name == 'div':
        return (1 + x / 3) / (x + mp.mpf(1) / 7)

worst = {}
for line in sys.stdin:
    name, x, hi, lo = line.split()
    r = ref(name, float(x))
    v = mp.mpf(float(hi)) + mp.mpf(float(lo))
    err = abs(v / r - 1) if r != 0 else abs(v)
    if err > worst.get(name, (0, 0))[0]:
        worst[name] = (err, x)

fail = False
for name, (err, x) in sorted(worst.items()):
    print(f'{name:5s} max rel err {mp.nstr(err, 3):>10s} at x = {x}')
    fail |= err > 1e-30
sys.exit(1 if fail else 0)
