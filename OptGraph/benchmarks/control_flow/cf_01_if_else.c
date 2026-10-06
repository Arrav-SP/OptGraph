#include <stdio.h>
/* Plain if/else chains selecting between arithmetic updates. */
static int classify(int v) {
    int r;
    if (v < 10) r = v * 2;
    else if (v < 30) r = v + 5;
    else if (v < 60) r = v - 7;
    else r = v / 3;
    return r;
}
int main(void) {
    int s = 0;
    for (int i = 0; i < 90; i++) {
        int c = classify(i);
        if (c % 2 == 0) s += c; else s -= 1;
    }
    printf("%d\n", s);
    return 0;
}