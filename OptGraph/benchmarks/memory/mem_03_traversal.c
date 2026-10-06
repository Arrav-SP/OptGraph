#include <stdio.h>
#define N 64
/* Index-linked list stored in arrays, traversed through pointers. */
struct node { int value; int next; };
static int walk(const struct node *nodes, int head) {
    int total = 0;
    const struct node *p = &nodes[head];
    for (;;) {
        total += p->value;
        if (p->next < 0) break;
        p = &nodes[p->next];
    }
    return total;
}
int main(void) {
    struct node nodes[N];
    for (int i = 0; i < N; i++) {
        nodes[i].value = (i * 37) % 101;
        nodes[i].next = (i + 7 < N) ? i + 7 : -1;
    }
    int t = 0;
    for (int h = 0; h < 7; h++) t += walk(nodes, h);
    printf("%d\n", t);
    return 0;
}