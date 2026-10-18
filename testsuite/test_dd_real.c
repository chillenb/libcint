/*
 * Prints dd_real.h results for exp, sqrt, erf, erfc and div on a grid
 * of arguments.  test_dd_real.py compares them against mpmath.
 *
 * gcc -O2 -ffp-contract=off -I../src test_dd_real.c -o test_dd_real -lm
 */
#include <stdio.h>
#include "dd_real.h"

static void out(const char *name, double x, dd_real r)
{
        printf("%s %.17e %.17e %.17e\n", name, x, r.hi, r.lo);
}

int main(void)
{
        int i;
        double x;
        for (i = 0; i <= 4000; i++) {
                x = i * 0.05;              // [0, 200]
                out("expm", x, dd_exp(dd_from_d(-x)));
        }
        for (i = 1; i <= 3000; i++) {
                x = i * 0.005;             // (0, 15]
                out("erf", x, dd_erf(dd_from_d(x)));
                out("erfc", x, dd_erfc(dd_from_d(x)));
        }
        for (i = 1; i <= 1000; i++) {
                x = i * 0.37;
                out("sqrt", x, dd_sqrt(dd_from_d(x)));
                // (1 + x/3) / (x + 1/7), both rounded to dd first
                dd_real a = dd_add_d(dd_div_d(dd_from_d(x), 3.), 1.);
                dd_real b = dd_add_d(dd_div_d(dd_from_d(1.), 7.), x);
                out("div", x, dd_div(a, b));
        }
        return 0;
}
