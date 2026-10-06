#include <stdio.h>
/* Small constant trip counts: candidates for rotation and full unrolling. */
static unsigned dot4(const unsigned *a, const unsigned *b) {
    unsigned s = 0u;
    for (int i = 0; i < 4; i++) s += a[i] * b[i];
    return s;
}
int main(void) {
    unsigned a[4] = {1u, 2u, 3u, 4u};
    unsigned b[4] = {5u, 6u, 7u, 8u};
    unsigned t = 0u;
    for (int r = 0; r < 3; r++) {
        t += dot4(a, b);
        for (int i = 0; i < 4; i++) a[i] += b[i];
    }
    printf("%u\n", t);
    return 0;
}