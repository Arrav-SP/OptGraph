#include <stdio.h>
/* Horner evaluation of a fixed polynomial at several points. */
static unsigned poly(unsigned x) {
    unsigned c[6] = {3u, 7u, 1u, 9u, 4u, 2u};
    unsigned r = 0u;
    for (int i = 5; i >= 0; i--) r = r * x + c[i];
    return r;
}
int main(void) {
    unsigned acc = 0u;
    for (unsigned x = 0u; x < 50u; x++) acc += poly(x) ^ (x * 3u);
    printf("%u\n", acc);
    return 0;
}