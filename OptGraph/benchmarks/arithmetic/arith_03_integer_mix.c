#include <stdio.h>
/* Integer helpers: gcd, modular exponentiation, bit counting. */
static unsigned gcd(unsigned a, unsigned b) {
    while (b != 0u) { unsigned t = a % b; a = b; b = t; }
    return a;
}
static unsigned modpow(unsigned base, unsigned e, unsigned m) {
    unsigned long long r = 1ull, b = base % m;
    while (e > 0u) {
        if (e & 1u) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1;
    }
    return (unsigned)r;
}
static unsigned popcount(unsigned v) {
    unsigned c = 0u;
    while (v) { c += v & 1u; v >>= 1; }
    return c;
}
int main(void) {
    unsigned s = 0u;
    for (unsigned i = 1u; i < 60u; i++)
        s += gcd(i * 12u, 90u) + modpow(i, 13u, 1009u) + popcount(i * 2654435761u);
    printf("%u\n", s);
    return 0;
}