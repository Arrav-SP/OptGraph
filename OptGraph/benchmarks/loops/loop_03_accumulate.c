#include <stdio.h>
/* Several accumulators updated by while and do-while loops. */
int main(void) {
    unsigned sum = 0u, prod = 1u, alt = 0u, i = 1u;
    while (i <= 100u) {
        sum += i;
        prod = (prod * (i | 1u)) % 65521u;
        if (i % 2u == 0u) alt += i; else alt -= i;
        i++;
    }
    unsigned n = 27u, steps = 0u;
    do {
        n = (n % 2u == 0u) ? n / 2u : 3u * n + 1u;
        steps++;
    } while (n != 1u);
    printf("%u %u %u %u\n", sum, prod, alt, steps);
    return 0;
}