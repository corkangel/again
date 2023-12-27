#include <iostream>
#include <random>
#include <cassert>
#include <array>

#include <fstream>
#include <iterator>

#include "model.h"
#include "render.h"

// each image is 32 x 32 x 3
const int imageArraySize = 32 * 32 * 3;
const int numImages = 400;
const int numCategories = 10;

// each image in the data is this many bytes
const int imageDataSize = 1 + 1024*3;

// constant to covert from 255 to float in 0-to-1 range
const double convert255 = double(1) / double(255);

void loadImages(const char* filename, matrix& images, matrix& categories)
{
    std::ifstream input(filename, std::ios::binary );
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

    // flatten and scale image data from the file into a matrix
    for (int i=0; i < numImages; i++)
    {
        images[i].resize(1024 * 3);

        unsigned char* imagePtr = &buffer[i*imageDataSize];
        unsigned char* r = &imagePtr[1];
        unsigned char* g = &imagePtr[1+1024];
        unsigned char* b = &imagePtr[1+1024+1024];

        for (int pixel = 0; pixel < 1024; pixel++)
        {
            const int o = pixel*3;
            images[i][o+0] = (*r++) * convert255;
            images[i][o+1] = (*g++) * convert255;
            images[i][o+2] = (*b++) * convert255;
        }

        // hot encode categories
        categories[i].resize(numCategories);
        unsigned char category = *imagePtr;
        for (int c=0; c < numCategories; c++)
        {
            categories[i][c] = (category == c) ? 1 : 0;
        }
    }
}

int argmax(const column& values)
{
    int index = 0;
    double highest = -1000;
    for(int i=0; i < values.size(); i++)
    {
        if (values[i] > highest)
        {
            highest = values[i];
            index = i;
        }
    }
    return index;
}

int main(int, char**)
{
    // stable random values
    srand(101010101);

    // need image data in matrix of [numImages][width][height][color], in 0.0 to 1.0 range

    matrix batch1Images(numImages);
    matrix batch1Categories(numImages);
    loadImages("Resources/Data/data_batch_1.bin", batch1Images, batch1Categories);
    
    
    matrix testImages(numImages);
    matrix testCategories(numImages);
    loadImages("Resources/Data/test_batch.bin", testImages, testCategories);
    

    model m;
    layer* l = m.AddInputLayer(imageArraySize); // input layer for one image
    l = m.AddDenseLayer(200, ActivationFunction::Relu, l);  
    l = m.AddDenseLayer(150, ActivationFunction::Relu, l); 
    l = m.AddDenseLayer(10, ActivationFunction::Softmax, l); 

    renderWindow rw;


    // stable random set of test indices
    const int numTests = 1;    
    std::array<int, numTests> testIds;
    for (int j=0; j< numTests; j++)
    testIds[j] = rand() % numTests;

    int trainingRuns = 0;
    bool running = 1;
    while (running)
    {
        m.Train(batch1Images, batch1Categories, 1, 0.1);

        trainingRuns++;
        {
            column predictions(numCategories);

            int numCorrect = 0;

            // calculate accuracy on a subset of test images
            for (int t=0; t < numTests; t++)
            {
                const int index = testIds[t];

                m.PredictSingleInput(testImages[index], predictions);
                
                int predictedCategory = argmax(predictions);
                int testCategory = argmax(testCategories[index]);

                if (predictedCategory == testCategory)
                    numCorrect += 1;

            }
            m.loss = double(numCorrect) / numTests;
        }

        rw.ProcessEvents(running);

        rw.BeginDisplay();
        rw.DisplayTitle(m.epoch, m.loss, "Images");



        // const int numImages = 20;
        // for (int y=0; y < numImages; y++)
        // {
        //     for (int x=0; x < numImages; x++)
        //     {
        //         const int iz = 1 + 1024*3;
        //         int o = y*numImages+x;
        //         unsigned char *r = &buffer[iz*o + 1];
        //         unsigned char *g = &buffer[iz*o + 1+(1024)];
        //         unsigned char *b = &buffer[iz*o + 1+(1024)+(1024)];
        //         rw.DisplayImage(&r[0], &g[0], &b[0], 20 + x*40, 160 + y*40);
        //     }
        // }
        
        rw.EndDisplay();
    }
}
