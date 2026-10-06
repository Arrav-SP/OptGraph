#include <stdio.h>
#define N 128
/* Loop bodies that recompute loop-invariant expressions each iteration. */
static unsigned scale(unsigned *out, const unsigned *in, unsigned a, unsigned b, int n) {
    unsigned s = 0u;
    for (int i = 0; i < n; i++) {
        unsigned k = a * b + (a << 2);
        unsigned m = (b + 17u) * (b + 17u);
        out[i] = in[i] * k + m;
        s += out[i];
    }
    return s;
}
int main(void) {
    unsigned in[N], out[N];
    for (int i = 0; i < N; i++) in[i] = (unsigned)i;
    unsigned s = scale(out, in, 5u, 9u, N);
    printf("%u\n", s);
    return 0;
}