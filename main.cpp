#include <iostream>
#include <random>

#include "model.h"
#include "render.h"

int main(int, char**)
{
    // stable random values
    srand(101010101);

    const matrix inputs = {
        {0.1, 0.1},
        {0.5, 0.5},
        {0.9, 0.8}
    };

    const matrix targets = {
        {.9, .1, .9},
        {.1, .9, .1},
        {.3, .1, .9}
    };

    model m;
    layer* l = m.AddLayer(2); // input layer
    l = m.AddLayer(3, l); // hiddenA
    l = m.AddLayer(6, l); // hiddenB
    l = m.AddLayer(3, l); // output layer

    renderWindow rw;

    // re-used for each prediction
    column ins(2); // (x,y)

     const int gridSize = 80;
     matrix outs(gridSize*gridSize);
     for (auto&& o : outs)
         o.resize(3, 0); // (r,g,b)
    
    bool running = 1;
    while (running)
    {
        m.Train(inputs, targets, 100, 0.001);

        for (int y=0; y < gridSize; y++)
        {
            for (int x=0; x < gridSize; x++)
            {
                ins[0] = x*1.0/gridSize;
                ins[1] = y*1.0/gridSize;
                m.PredictSingleInput(ins, outs[y*gridSize+x]);                
            }
        }

        rw.ProcessEvents(running);
        rw.Display(m.epoch, m.loss, gridSize, outs);
    }
}
