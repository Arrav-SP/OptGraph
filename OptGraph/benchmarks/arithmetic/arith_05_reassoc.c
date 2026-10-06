#include <stdio.h>
/* Constant chains that only fold after reassociation. */
static unsigned g(unsigned a, unsigned b, unsigned c) {
    unsigned x = (a + 5u) + (b + 7u);
    unsigned y = (c + 11u) + (a + 13u);
    unsigned z = (x * 2u) * 3u;
    unsigned w = ((a + b) + c) - ((c + b) + a);
    unsigned v = (4u * a) + (b * 4u);
    return x + y + z + w + v - 36u;
}
int main(void) {
    unsigned s = 0u;
    for (unsigned i = 0u; i < 30u; i++) s ^= g(i, i * 2u, i + 9u) + i;
    printf("%u\n", s);
    return 0;
}