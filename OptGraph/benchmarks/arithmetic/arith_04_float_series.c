#include <stdio.h>
/* Floating point: partial sums of two series and a Newton iteration. */
static double newton_sqrt(double v) {
    double x = v * 0.5 + 0.5;
    for (int i = 0; i < 12; i++) x = 0.5 * (x + v / x);
    return x;
}
int main(void) {
    double leibniz = 0.0, basel = 0.0, sign = 1.0;
    for (int k = 0; k < 400; k++) {
        double d = 2.0 * (double)k + 1.0;
        leibniz += sign / d;
        sign = -sign;
        double n = (double)(k + 1);
        basel += 1.0 / (n * n);
    }
    double r = newton_sqrt(basel * 6.0) + 4.0 * leibniz;
    printf("%.6f\n", r);
    return 0;
}