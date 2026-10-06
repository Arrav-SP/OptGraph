#include <stdio.h>
/* Redundant and identity arithmetic: x+0, x*1, repeated subexpressions. */
static unsigned f(unsigned a, unsigned b) {
    unsigned t1 = a + 0u;
    unsigned t2 = b * 1u;
    unsigned t3 = (a + b) * (a + b);
    unsigned t4 = (a + b) * (a + b);
    unsigned t5 = t1 - t1 + t2;
    unsigned t6 = (t3 + t4) / 2u;
    unsigned t7 = t6 * 4u;
    unsigned t8 = (a ^ a) | t5;
    return t7 + t8 + (t2 << 0);
}
int main(void) {
    unsigned s = 0u;
    for (unsigned i = 1u; i < 40u; i++) s += f(i, i + 3u);
    printf("%u\n", s);
    return 0;
}