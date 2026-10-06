#include <stdio.h>
/* Triple nested counting loops. */
int main(void) {
    unsigned s = 0u;
    for (unsigned i = 0u; i < 20u; i++)
        for (unsigned j = 0u; j < 15u; j++)
            for (unsigned k = 0u; k < 10u; k++)
                s += (i * j) ^ (k + i);
    printf("%u\n", s);
    return 0;
}