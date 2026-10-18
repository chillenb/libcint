/*
 * Double-double arithmetic for the extended-precision Rys quadrature
 * (enabled by the CMake option WITH_DOUBLE_DOUBLE).
 *
 * A dd_real value is the unevaluated sum hi + lo of two doubles with
 * |lo| <= ulp(hi)/2, giving about 106 bits of significand.
 *
 * The arithmetic kernels, sqrt and exp are a C port of the QD library
 *   https://github.com/BL-highprecision/QD  (commit b527a4b)
 * from the files include/qd/inline.h, include/qd/dd_inline.h,
 * src/dd_real.cpp and src/dd_const.cpp.  The algorithms and constant
 * tables are kept as in QD, except that the argument reduction in
 * dd_exp is done with exact products and a three-word log(2).  The port
 * uses the QD_IEEE_ADD variant of addition and the accurate (non-sloppy)
 * division.  dd_erf and dd_erfc (using coefficients from D. N. Laikov,
 * arXiv:2504.05068, see below)
 * are not part of QD and were written for libcint.
 *
 * Correctness requires IEEE double arithmetic with round-to-nearest,
 * no extended-precision intermediates, no FMA contraction
 * (-ffp-contract=off) and no -ffast-math.
 *
 * ---------------------------------------------------------------------
 * The QD files carry the notice:
 *
 *   This work was supported by the Director, Office of Science, Division
 *   of Mathematical, Information, and Computational Sciences of the
 *   U.S. Department of Energy under contract number DE-AC03-76SF00098.
 *
 *   Copyright (c) 2000-2007
 *
 * and are distributed under the following license:
 *
 *   BSD 3-Clause License
 *
 *   Copyright (c) 2021, highprecision
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *
 *   3. Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */

#ifndef HAVE_DEFINED_DD_REAL_H
#define HAVE_DEFINED_DD_REAL_H

#include <stdio.h>
#include <math.h>

typedef struct {
        double hi;
        double lo;
} dd_real;

static const dd_real DD_LOG2 = {6.931471805599452862e-01, 2.319046813846299558e-17};
// third word of log(2): log(2) - DD_LOG2.hi - DD_LOG2.lo
#define DD_LOG2_LO2     5.70770843841621207e-34
// sqrt(pi)/2
static const dd_real DD_SQRTPIE4 = {8.86226925452758052e-01, -3.83329324991289935e-17};
// 1/sqrt(pi)
static const dd_real DD_SQRTPI_INV = {5.64189583547756279e-01, 7.66772980658294061e-18};
#define DD_EPS  4.93038065763132e-32  // 2^-104

static inline dd_real dd_make(double hi, double lo)
{
        dd_real r = {hi, lo};
        return r;
}

static inline dd_real dd_from_d(double a)
{
        dd_real r = {a, 0.};
        return r;
}

static inline double dd_to_d(dd_real a)
{
        return a.hi;
}

/*********** Error-free transformations (qd/inline.h) ************/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
static inline double dd_quick_two_sum(double a, double b, double *err)
{
        double s = a + b;
        *err = b - (s - a);
        return s;
}

/* Computes fl(a+b) and err(a+b). */
static inline double dd_two_sum(double a, double b, double *err)
{
        double s = a + b;
        double bb = s - a;
        *err = (a - (s - bb)) + (b - bb);
        return s;
}

/* Computes fl(a-b) and err(a-b). */
static inline double dd_two_diff(double a, double b, double *err)
{
        double s = a - b;
        double bb = s - a;
        *err = (a - (s - bb)) - (b + bb);
        return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline double dd_two_prod(double a, double b, double *err)
{
        double p = a * b;
        *err = fma(a, b, -p);
        return p;
}

/* Computes fl(a*a) and err(a*a). */
static inline double dd_two_sqr(double a, double *err)
{
        double p = a * a;
        *err = fma(a, a, -p);
        return p;
}

/*********** Additions (qd/dd_inline.h) ************/
/* double-double = double + double */
static inline dd_real dd_add_dd(double a, double b)
{
        double s, e;
        s = dd_two_sum(a, b, &e);
        return dd_make(s, e);
}

/* double-double + double */
static inline dd_real dd_add_d(dd_real a, double b)
{
        double s1, s2;
        s1 = dd_two_sum(a.hi, b, &s2);
        s2 += a.lo;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        return dd_make(s1, s2);
}

/* double-double + double-double, IEEE-style error bound (Briggs, Kahan) */
static inline dd_real dd_add(dd_real a, dd_real b)
{
#ifdef DD_SLOPPY_ADD
        double s, e;
        s = dd_two_sum(a.hi, b.hi, &e);
        e += (a.lo + b.lo);
        s = dd_quick_two_sum(s, e, &e);
        return dd_make(s, e);
#else
        double s1, s2, t1, t2;
        s1 = dd_two_sum(a.hi, b.hi, &s2);
        t1 = dd_two_sum(a.lo, b.lo, &t2);
        s2 += t1;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        s2 += t2;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        return dd_make(s1, s2);
#endif
}

/* double-double - double */
static inline dd_real dd_sub_d(dd_real a, double b)
{
        double s1, s2;
        s1 = dd_two_diff(a.hi, b, &s2);
        s2 += a.lo;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        return dd_make(s1, s2);
}

/* double - double-double */
static inline dd_real dd_d_sub(double a, dd_real b)
{
        double s1, s2;
        s1 = dd_two_diff(a, b.hi, &s2);
        s2 -= b.lo;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        return dd_make(s1, s2);
}

/* double-double - double-double */
static inline dd_real dd_sub(dd_real a, dd_real b)
{
#ifdef DD_SLOPPY_ADD
        double s, e;
        s = dd_two_diff(a.hi, b.hi, &e);
        e += a.lo;
        e -= b.lo;
        s = dd_quick_two_sum(s, e, &e);
        return dd_make(s, e);
#else
        double s1, s2, t1, t2;
        s1 = dd_two_diff(a.hi, b.hi, &s2);
        t1 = dd_two_diff(a.lo, b.lo, &t2);
        s2 += t1;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        s2 += t2;
        s1 = dd_quick_two_sum(s1, s2, &s2);
        return dd_make(s1, s2);
#endif
}

static inline dd_real dd_neg(dd_real a)
{
        return dd_make(-a.hi, -a.lo);
}

static inline dd_real dd_fabs(dd_real a)
{
        return (a.hi < 0.) ? dd_neg(a) : a;
}

/*********** Multiplications ************/
/* double-double = double * double */
static inline dd_real dd_mul_dd(double a, double b)
{
        double p, e;
        p = dd_two_prod(a, b, &e);
        return dd_make(p, e);
}

/* double-double * (2.0 ^ exp) */
static inline dd_real dd_ldexp(dd_real a, int exp)
{
        return dd_make(ldexp(a.hi, exp), ldexp(a.lo, exp));
}

/* double-double * double,  where double is a power of 2. */
static inline dd_real dd_mul_pwr2(dd_real a, double b)
{
        return dd_make(a.hi * b, a.lo * b);
}

/* double-double * double */
static inline dd_real dd_mul_d(dd_real a, double b)
{
        double p1, p2;
        p1 = dd_two_prod(a.hi, b, &p2);
        p2 += (a.lo * b);
        p1 = dd_quick_two_sum(p1, p2, &p2);
        return dd_make(p1, p2);
}

/* double-double * double-double */
static inline dd_real dd_mul(dd_real a, dd_real b)
{
        double p1, p2;
        p1 = dd_two_prod(a.hi, b.hi, &p2);
        p2 += (a.hi * b.lo + a.lo * b.hi);
        p1 = dd_quick_two_sum(p1, p2, &p2);
        return dd_make(p1, p2);
}

static inline dd_real dd_sqr(dd_real a)
{
        double p1, p2;
        double s1, s2;
        p1 = dd_two_sqr(a.hi, &p2);
        p2 += 2.0 * a.hi * a.lo;
        p2 += a.lo * a.lo;
        s1 = dd_quick_two_sum(p1, p2, &s2);
        return dd_make(s1, s2);
}

/*********** Divisions ************/
/* Rescale very large numerators by 2^-53 before quotient reconstruction
   so intermediate products such as q1 * b do not overflow in double. */
static inline int dd_div_needs_rescale(double a_hi)
{
        return fabs(a_hi) > 0x1p+969;
}

/* double-double / double */
static inline dd_real dd_div_d(dd_real a, double b)
{
        double q1, q2;
        double p1, p2;
        double s, e;
        dd_real r;

        const int rescale = dd_div_needs_rescale(a.hi);
        const dd_real aa = rescale ? dd_mul_pwr2(a, 0x1p-53) : a;

        q1 = aa.hi / b;   /* approximate quotient. */

        /* Compute  this - q1 * d */
        p1 = dd_two_prod(q1, b, &p2);
        s = dd_two_diff(aa.hi, p1, &e);
        e += aa.lo;
        e -= p2;

        /* get next approximation. */
        q2 = (s + e) / b;

        /* renormalize */
        r.hi = dd_quick_two_sum(q1, q2, &r.lo);

        return rescale ? dd_mul_pwr2(r, 0x1p+53) : r;
}

/* double-double / double-double (QD accurate_div) */
static inline dd_real dd_div(dd_real a, dd_real b)
{
#ifdef DD_SLOPPY_DIV
        double s1, s2;
        double q1, q2;
        dd_real r;

        const int rescale = dd_div_needs_rescale(a.hi);
        const dd_real aa = rescale ? dd_mul_pwr2(a, 0x1p-53) : a;

        q1 = aa.hi / b.hi;  /* approximate quotient */

        /* compute  this - q1 * dd */
        r = dd_mul_d(b, q1);
        s1 = dd_two_diff(aa.hi, r.hi, &s2);
        s2 -= r.lo;
        s2 += aa.lo;

        /* get next approximation */
        q2 = (s1 + s2) / b.hi;

        /* renormalize */
        r.hi = dd_quick_two_sum(q1, q2, &r.lo);
        return rescale ? dd_mul_pwr2(r, 0x1p+53) : r;
#else
        double q1, q2, q3;
        dd_real r;

        const int rescale = dd_div_needs_rescale(a.hi);
        const dd_real aa = rescale ? dd_mul_pwr2(a, 0x1p-53) : a;

        q1 = aa.hi / b.hi;  /* approximate quotient */

        r = dd_sub(aa, dd_mul_d(b, q1));

        q2 = r.hi / b.hi;
        r = dd_sub(r, dd_mul_d(b, q2));

        q3 = r.hi / b.hi;

        q1 = dd_quick_two_sum(q1, q2, &q2);
        r = dd_add_d(dd_make(q1, q2), q3);
        return rescale ? dd_mul_pwr2(r, 0x1p+53) : r;
#endif
}

/* double / double-double */
static inline dd_real dd_d_div(double a, dd_real b)
{
        return dd_div(dd_from_d(a), b);
}

/*********** Comparisons (operands are normalized) ************/
static inline int dd_lt(dd_real a, dd_real b)
{
        return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

static inline int dd_lt_d(dd_real a, double b)
{
        return a.hi < b || (a.hi == b && a.lo < 0.);
}

static inline int dd_gt_d(dd_real a, double b)
{
        return a.hi > b || (a.hi == b && a.lo > 0.);
}

static inline int dd_le_d(dd_real a, double b)
{
        return !dd_gt_d(a, b);
}

/*********** sqrt and exp (src/dd_real.cpp) ************/
static inline dd_real dd_sqrt_karp(dd_real a)
{
        double x = 1.0 / sqrt(a.hi);
        double ax = a.hi * x;
        return dd_add_dd(ax, dd_sub(a, dd_mul_dd(ax, ax)).hi * (x * 0.5));
}

/* Computes the square root of the double-double number a.
   Karp's trick: if x is an approximation to 1/sqrt(a), then
        sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)  */
static inline dd_real dd_sqrt(dd_real a)
{
        if (a.hi == 0.) {
                return dd_from_d(0.);
        }
        if (a.hi < 0.) {
                fprintf(stderr, "libcint::dd_sqrt negative argument %g\n", a.hi);
                return dd_make(NAN, NAN);
        }
        /* Avoid overflow in the Karp refinement for extremely large inputs by
           using the exact identity sqrt(a) = 2 * sqrt(a / 4). */
        if (a.hi > 0x1p+969) {
                return dd_mul_pwr2(dd_sqrt_karp(dd_mul_pwr2(a, 0.25)), 2.0);
        }
        return dd_sqrt_karp(a);
}

static const double DD_INV_FACT[15][2] = {
        { 1.66666666666666657e-01,  9.25185853854297066e-18},
        { 4.16666666666666644e-02,  2.31296463463574266e-18},
        { 8.33333333333333322e-03,  1.15648231731787138e-19},
        { 1.38888888888888894e-03, -5.30054395437357706e-20},
        { 1.98412698412698413e-04,  1.72095582934207053e-22},
        { 2.48015873015873016e-05,  2.15119478667758816e-23},
        { 2.75573192239858925e-06, -1.85839327404647208e-22},
        { 2.75573192239858883e-07,  2.37677146222502973e-23},
        { 2.50521083854417202e-08, -1.44881407093591197e-24},
        { 2.08767569878681002e-09, -1.20734505911325997e-25},
        { 1.60590438368216133e-10,  1.25852945887520981e-26},
        { 1.14707455977297245e-11,  2.06555127528307454e-28},
        { 7.64716373181981641e-13,  7.03872877733453001e-30},
        { 4.77947733238738525e-14,  4.39920548583408126e-31},
        { 2.81145725434552060e-15,  1.65088427308614326e-31}
};

/* Exponential.  We first reduce the size of x by noting that
        exp(kr + m * log(2)) = 2^m * exp(r)^k
   where m and k are integers.  By choosing m appropriately
   we can make |kr| <= log(2) / 2 = 0.347.  Then exp(r) is
   evaluated using the familiar Taylor series.  */
static inline dd_real dd_exp(dd_real a)
{
        const double k = 512.0;
        const double inv_k = 1.0 / k;

        if (a.hi <= -709.0) {
                return dd_from_d(0.);
        }
        if (a.hi >= 709.0) {
                return dd_make(INFINITY, INFINITY);
        }
        if (a.hi == 0.) {
                return dd_from_d(1.);
        }

        double m = floor(a.hi / DD_LOG2.hi + 0.5);
        /* libcint modification: QD computes a - log(2) * m with dd * double,
           which rounds log(2).lo * m and makes the relative error of exp
           grow like |a| * 1e-32 (2e-30 at a = -140).  Here the products
           m * log(2).hi and m * log(2).lo are formed exactly and a third
           word of log(2) is included. */
        dd_real r = dd_sub(a, dd_mul_dd(m, DD_LOG2.hi));
        r = dd_sub(r, dd_mul_dd(m, DD_LOG2.lo));
        r = dd_mul_pwr2(dd_sub_d(r, m * DD_LOG2_LO2), inv_k);
        dd_real s, t, p;

        p = dd_sqr(r);
        s = dd_add(r, dd_mul_pwr2(p, 0.5));
        p = dd_mul(p, r);
        t = dd_mul(p, dd_make(DD_INV_FACT[0][0], DD_INV_FACT[0][1]));
        int i = 0;
        do {
                s = dd_add(s, t);
                p = dd_mul(p, r);
                ++i;
                t = dd_mul(p, dd_make(DD_INV_FACT[i][0], DD_INV_FACT[i][1]));
        } while (fabs(t.hi) > inv_k * DD_EPS && i < 5);

        s = dd_add(s, t);

        for (i = 0; i < 9; i++) {
                s = dd_add(dd_mul_pwr2(s, 2.0), dd_sqr(s));
        }
        s = dd_add_d(s, 1.0);

        return dd_ldexp(s, (int)m);
}

/*********** erf and erfc (not part of QD) ************/
/*
 * erf for x < 1 is evaluated with the series of positive terms
 *      erf(x) = 2/sqrt(pi) exp(-x^2) sum_{k>=0} (2x^2)^k x / (2k+1)!!
 *
 * erfc is evaluated with the approximation of D. N. Laikov,
 * "Global approximations to the error function of real argument for
 * vectorized computation", arXiv:2504.05068v4, Eq. (18):
 *      erfc(x) ~ (1 + sum_{m=1}^N C_m x^m) / (1 + sum_{n=1}^{N+1} D_n x^n) exp(-x^2)
 * with the N = 20 coefficients optimized for uniform relative accuracy
 * (110.3 bits, Table 1; function ferfc(v20) of the ancillary file ferf_q.c),
 * split into double-double pairs.  All coefficients are positive, so the
 * Horner evaluation for x >= 0 has no cancellation.
 * erf = 1 - erfc for x >= 1.  Arguments are assumed to be >= 0.
 *
 * The coefficients are used with the agreement of Dr. Laikov (2026), who
 * asks that the published article be cited.  The article is expected to
 * appear in the Journal of Chemical Physics; once it does, replace the
 * reference arXiv:2504.05068 above with the journal citation.
 *
 * With DD_ERFC_CONTINUED_FRACTION defined, erfc for x >= 1 is instead
 * evaluated from the continued fraction
 *      erfc(x) = x exp(-x^2)/sqrt(pi) /
 *                (x^2+1/2 - (1*1/2)/(x^2+5/2 - (2*3/2)/(x^2+9/2 - ...)))
 * in backward recurrence with n = 360/x^2 + 12 terms (|rel. err| < 1e-32
 * empirically, but up to ~370 divisions near x = 1), and for x < 1 as
 * 1 - erf.
 */
static inline dd_real dd_erf_series(dd_real x)
{
        dd_real x2 = dd_sqr(x);
        dd_real z = dd_mul_pwr2(x2, 2.);
        dd_real term = x;
        dd_real s = x;
        int k;
        for (k = 1; k < 200; k++) {
                term = dd_div_d(dd_mul(term, z), (double)(2 * k + 1));
                s = dd_add(s, term);
                if (term.hi < s.hi * DD_EPS * .25) {
                        break;
                }
        }
        // 2/sqrt(pi) * exp(-x^2) * s
        s = dd_mul(dd_mul_pwr2(DD_SQRTPI_INV, 2.), dd_mul(dd_exp(dd_neg(x2)), s));
        return s;
}

static inline dd_real dd_erfc_cf(dd_real x)
{
        dd_real x2 = dd_sqr(x);
        int n = (int)(360. / x2.hi) + 12;
        int k;
        dd_real f = dd_add_d(x2, 2 * n + .5);
        for (k = n; k > 0; k--) {
                f = dd_sub(dd_add_d(x2, 2 * (k - 1) + .5),
                           dd_d_div(k * (k - .5), f));
        }
        dd_real e = dd_exp(dd_neg(x2));
        return dd_div(dd_mul(dd_mul(x, e), DD_SQRTPI_INV), f);
}

static const double DD_ERFC_V20[42][2] = {
        {0x1.b7ae63e87eb60p+5, 0x1.a292232baf493p-49},
        {0x1.20dd750429b6dp-1, 0x1.1ae3a914fed80p-57},
        {0x1.78f8d29b097d3p+10, 0x1.6cf0453c72dbep-44},
        {0x1.f0208aeba849ep+4, 0x1.ca4d2c9b68f07p-50},
        {0x1.aae195e2cbf0bp+14, 0x1.0bd541330ad55p-40},
        {0x1.a939ec5d73901p+9, -0x1.42b8d175bc492p-45},
        {0x1.64a909b399fa1p+18, -0x1.c46f0f6f93987p-37},
        {0x1.e133085ad9e4cp+13, -0x1.3f3f085003f6cp-46},
        {0x1.d20c6bb7ef72fp+21, -0x1.3724391941079p-34},
        {0x1.919e3b03fb5d0p+17, -0x1.6c01cf8f9b88cp-37},
        {0x1.ecffedac42bc1p+24, 0x1.404a351e7d6a0p-32},
        {0x1.060073fb8b924p+21, -0x1.0e7913296be20p-36},
        {0x1.af8fa971cd7aep+27, 0x1.7a8f02474b15ap-28},
        {0x1.14953d1fd857dp+24, -0x1.2c05e3025908cp-30},
        {0x1.3d186c2d17b97p+30, 0x1.be0487a3297e0p-24},
        {0x1.e2e66c348bf10p+26, 0x1.1683e54f4fd92p-29},
        {0x1.8aa43e59e49fbp+32, 0x1.9814ef14c52bep-23},
        {0x1.6187df932163fp+29, -0x1.a72cebbe57d32p-26},
        {0x1.a20c32914b1bbp+34, 0x1.e161d5dfb32fbp-20},
        {0x1.b5e2bb6efda1dp+31, 0x1.d75b69f08ec74p-25},
        {0x1.7998f411079f7p+36, 0x1.44ff0ddfb5ca5p-18},
        {0x1.ccee5476d31e7p+33, -0x1.e73b23ce3abfep-21},
        {0x1.22793ef45aef0p+38, 0x1.47eb0334ddd3ep-16},
        {0x1.9cd76064e4889p+35, -0x1.870b5dda9b4f9p-19},
        {0x1.7b0cbec509e34p+39, 0x1.b259a996848e1p-16},
        {0x1.3a02a38bb5c5cp+37, -0x1.163f2fede4a9ep-17},
        {0x1.a06da84d3259cp+40, 0x1.0b6d95c2c8624p-18},
        {0x1.937b39f8a0292p+38, -0x1.46406e7783f24p-19},
        {0x1.7cc1fab964169p+41, -0x1.e4062a8f8d05ap-14},
        {0x1.b1d81f58d2848p+39, -0x1.b4fcb1de5e6b2p-20},
        {0x1.1ce13ff5de4a3p+42, 0x1.f596d68e0574ep-12},
        {0x1.80b614c64b21dp+40, 0x1.8d071bacf3f3bp-14},
        {0x1.5439175c3818bp+42, -0x1.88643d6e96bc4p-12},
        {0x1.1323c8a1da108p+41, -0x1.8b17173c73acdp-17},
        {0x1.382fdfeec31b3p+42, -0x1.44d73ebdbbee0p-12},
        {0x1.328f5b6696f6bp+41, -0x1.d1cde9e3ed172p-13},
        {0x1.9db56ec549786p+41, -0x1.4dec554dc3a47p-13},
        {0x1.f621de02102b2p+40, 0x1.d57cd750678ffp-14},
        {0x1.60ae04ce90513p+40, 0x1.2a591435538dap-16},
        {0x1.0eadce3f949e1p+40, -0x1.bbd0b687b64acp-14},
        {0x1.22af753d292a5p+38, 0x1.d6cb7cd600f5cp-17},
        {0x1.22af753d292a5p+38, 0x1.d6cb7cd600f5cp-17},
};

static inline dd_real dd_erfc_v20(dd_real x)
{
        const double (*c)[2] = DD_ERFC_V20;
        dd_real q = dd_add(x, dd_make(c[0][0], c[0][1]));
        dd_real p = dd_make(c[1][0], c[1][1]);
        int k;
        for (k = 2; k <= 40; k += 2) {
                q = dd_add(dd_mul(q, x), dd_make(c[k][0], c[k][1]));
                p = dd_add(dd_mul(p, x), dd_make(c[k+1][0], c[k+1][1]));
        }
        return dd_mul(dd_div(p, q), dd_exp(dd_neg(dd_sqr(x))));
}

static inline dd_real dd_erf(dd_real x)
{
        if (x.hi == 0.) {
                return dd_from_d(0.);
        } else if (x.hi < 1.) {
                return dd_erf_series(x);
        } else if (x.hi > 27.) {
                // erfc(27) < 1e-318
                return dd_from_d(1.);
        } else {
#ifdef DD_ERFC_CONTINUED_FRACTION
                return dd_d_sub(1., dd_erfc_cf(x));
#else
                return dd_d_sub(1., dd_erfc_v20(x));
#endif
        }
}

static inline dd_real dd_erfc(dd_real x)
{
        if (x.hi > 27.) {
                return dd_from_d(0.);
        }
#ifdef DD_ERFC_CONTINUED_FRACTION
        if (x.hi < 1.) {
                return dd_d_sub(1., dd_erf_series(x));
        }
        return dd_erfc_cf(x);
#else
        return dd_erfc_v20(x);
#endif
}

#endif
