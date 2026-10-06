#include <stdio.h>
/* Many small helper functions, including recursion. */
static int sq(int x) { return x * x; }
static int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
static int mix(int a, int b) { return clamp(sq(a) - b, -50, 500) + clamp(b, 0, 9); }
static int apply(int (*fn)(int), int v) { return fn(v) + 1; }
int main(void) {
    int s = 0;
    for (int i = 0; i < 20; i++) s += mix(i, fib(i % 12)) + apply(sq, i % 5);
    printf("%d\n", s);
    return 0;
}