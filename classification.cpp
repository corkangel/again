#include <cassert>

#include "model.h"

#pragma warning( disable : 4996 )

const uint32 numCategories = 10;
const double loadFactor = double(1)/16;

void loadFile(const char* filename, matrix& output, const uint32 columns, const uint32 rows, const double factor)
{
    output.resize(rows);

    FILE *fp = fopen(filename,"r");

    for(uint32 r=0; r < rows; r++)
    {
        output[r].resize(columns);
        for (uint32 c = 0; c < columns; c++)
        {
            int value;
            fscanf(fp, "%d ", &value);
            output[r][c] = double(value * factor);
        }
    }
    fclose(fp);
}

int main(int, char**)
{
    matrix allInputs;
    matrix allOutputs;
    matrix test_inputs;
    matrix test_outputs;

    std::vector<int>train_classes_outputs; 
    std::vector<int>test_classes_outputs;

    //3823 
    loadFile("Resources/Data/train_features.txt", allInputs, 64, 200, 1);
    loadFile("Resources/Data/train_output.txt", allOutputs, 1, 200, 1);
    
    //1797
    loadFile("Resources/Data/train_features.txt", test_inputs, 64, 200, 1);
    loadFile("Resources/Data/train_output.txt", test_outputs, 1, 200, 1);

    matrix hotEncodedOutputs(allOutputs.size());
    for (uint32 r = 0; r < allOutputs.size(); r++)
    {
        hotEncodedOutputs[r].resize(numCategories, 0);
        hotEncodedOutputs[r][ uint32(allOutputs[r][0]) ] = 1;
    }

    model m;
    layer* l = m.AddInputLayer(64);
    l = m.AddDenseLayer(200, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(100, ActivationFunction::Relu, l);
    l = m.AddDenseLayer(10, ActivationFunction::Softmax, l);

    for (int i = 0; i < 20; i++)
    {
        m.Train(allInputs, hotEncodedOutputs, 1, 0.1);
        printf("loss: %f\n", m.loss);
    }
    
    return 1;
}