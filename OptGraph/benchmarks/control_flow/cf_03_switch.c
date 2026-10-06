#include <stdio.h>
/* A small opcode interpreter built on a switch. */
static int step(int op, int acc, int arg) {
    switch (op) {
        case 0: return acc + arg;
        case 1: return acc - arg;
        case 2: return acc ^ arg;
        case 3: return (acc & 0xff) * 3;
        case 4: return acc | (arg << 1);
        case 5: return arg;
        default: return acc;
    }
}
int main(void) {
    int acc = 1;
    for (int i = 0; i < 200; i++) acc = step(i % 7, acc, i) & 0xffff;
    printf("%d\n", acc);
    return 0;
}