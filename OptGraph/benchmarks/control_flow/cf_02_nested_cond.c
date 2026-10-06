#include <stdio.h>
/* Deeply nested conditions with short-circuit operators. */
static int score(int a, int b, int c) {
    int r = 0;
    if (a > 3) {
        if (b > 2 && c < 8) {
            if ((a + b) % 3 == 0 || c == 1) r = 11; else r = 7;
        } else {
            if (b == c) r = 5; else r = (a > b) ? 3 : 2;
        }
    } else {
        if (b < 1 || (c > 4 && a == 2)) r = 1; else r = -1;
    }
    return r;
}
int main(void) {
    int s = 0;
    for (int a = 0; a < 8; a++)
        for (int b = 0; b < 6; b++)
            for (int c = 0; c < 10; c++) s += score(a, b, c);
    printf("%d\n", s);
    return 0;
}