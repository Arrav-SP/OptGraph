#include <stdio.h>
/* Branches whose conditions are compile-time constants, plus dead stores. */
static int compute(int v) {
    const int debug = 0;
    const int mode = 2;
    int unused = v * 31;
    int r = v;
    if (debug) { r = r * 1000; unused += 4; }
    if (mode == 1) r += 100;
    else if (mode == 2) r += 20;
    else r -= 5;
    if (sizeof(int) == 4) r ^= 0x55; else r ^= 0xaa;
    unused = unused + r;
    return r;
}
int main(void) {
    int s = 0;
    for (int i = 0; i < 64; i++) s += compute(i);
    printf("%d\n", s);
    return 0;
}