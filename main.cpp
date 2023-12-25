#include <iostream>
#include <random>

#include "model.h"
#include "render.h"

int main(int, char**)
{
    // stable random values
    srand(101010101);

    const matrix inputs = {
        {0.1, 0.3},
        {.3, .6},
        {.2, .8}
    };

    const matrix targets = {
        {.1,.2,.1},
        {.3,.3,.3},
        {.5,.5,.8}
    };

    model m;
    layer* l = m.AddLayer(2); // input layer
    l = m.AddLayer(3, l); // hiddenA
    l = m.AddLayer(2, l); // hiddenB
    l = m.AddLayer(3, l); // output layer

    renderWindow rw;

    bool running = 1;
    while (running)
    {
        m.Train(inputs, targets, 100);

        rw.ProcessEvents(running);
        rw.Display(m.loss);
    }
}
