#include <iostream>
#include <random>

#include "model.h"
#include "render.h"

int main(int, char**)
{
    // stable random values
    srand(101010101);

    const matrix inputs = {
        {.1, .1},
        {.9, .1},
        {.9, .9},
        {.1, .9},
        {.1, .5},
        {.9, .5},
        {.5, .1},
        {.5, .9},                
        {.3, .2},
        {.7, .2},
        {.3, .7},
        {.7, .7},             
    };

    const matrix targets = {
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},         
        {.95, .1, .1},                                  
        {.1, .99, .1},
        {.1, .99, .1},        
        {.1, .1, .97},                 
        {.1, .1, .97},                         
    };

    model m;
    layer* l = m.AddInputLayer(2); // input layer (x,y)
    l = m.AddDenseLayer(8, ActivationFunction::Sigmoid, l); // hiddenB
    l = m.AddDenseLayer(3, ActivationFunction::Sigmoid, l); // output layer (r,g,b)

    renderWindow rw;

     const int gridSize = 80;
     matrix outs(gridSize*gridSize);
     for (auto&& o : outs)
         o.resize(3, 0); // (r,g,b)
    
    bool running = 1;
    while (running)
    {
        m.Train(inputs, targets, 100, 0.1);

        for (int y=0; y < gridSize; y++)
        {
            for (int x=0; x < gridSize; x++)
            {
                column ins(2); // (x,y)
                ins[0] = x*1.0/gridSize;
                ins[1] = y*1.0/gridSize;
                m.PredictSingleInput(ins, outs[y*gridSize+x]);                
            }
        }

        column tmp1(3);
        column tmp2(3);
        column tmp3(3);
        m.PredictSingleInput(inputs[0], tmp1);       
        m.PredictSingleInput(inputs[1], tmp2);       
        m.PredictSingleInput(inputs[2], tmp3);       

        rw.ProcessEvents(running);

        rw.BeginDisplay();
        rw.DisplayTitle(m.epoch, m.loss, "Again");
        rw.DisplayGrid(
            m.layers.back()->gradients, 
            tmp1, tmp2, tmp3,
            gridSize, 
            outs);
        rw.EndDisplay();
    }
}
