#include <iostream>
#include <random>
#include <cassert>

#include <fstream>
#include <iterator>

#include "model.h"
#include "render.h"

#include <windows.h>

void ReadAll(unsigned char* buf, FILE* f, int sz)
{
    size_t pos = 0;
    while (pos < sz)
    {
        size_t ret = fread(&buf[pos], 1, sz-pos, f);
        if (ret)
        {
            pos += ret;
        }
        else
        {
            int e = feof(f);
            int n = ferror(f);
            n++;
        }
    }
}

int main(int, char**)
{
    // stable random values
    srand(101010101);

    TCHAR tszBuffer[1024];
    DWORD dwRet;

    dwRet = GetCurrentDirectory( MAX_PATH, tszBuffer );


    std::ifstream input("Resources/Data/data_batch_1.bin", std::ios::binary );
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});


    model m;
    layer* l = m.AddInputLayer(2); // input layer (x,y)
    l = m.AddDenseLayer(1, ActivationFunction::Sigmoid, l); // hiddenB
    l = m.AddDenseLayer(3, ActivationFunction::Sigmoid, l); // output layer (r,g,b)

    renderWindow rw;


    bool running = 1;
    while (running)
    {
        //m.Train(inputs, targets, 100, 0.1);

        rw.ProcessEvents(running);

        rw.BeginDisplay();
        rw.DisplayTitle(m.epoch, m.loss);

        const int numImages = 20;
        for (int y=0; y < numImages; y++)
        {
            for (int x=0; x < numImages; x++)
            {
                const int iz = 1 + 1024*3;
                int o = y*numImages+x;
                unsigned char *r = &buffer[iz*o + 1];
                unsigned char *g = &buffer[iz*o + 1+(1024)];
                unsigned char *b = &buffer[iz*o + 1+(1024)+(1024)];
                rw.DisplayImage(&r[0], &g[0], &b[0], 20 + x*40, 160 + y*40);
            }
        }
        
        rw.EndDisplay();
    }
}
