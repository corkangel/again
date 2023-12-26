#pragma once

#include "utils.h"

struct layer
{
    layer(int numNeurons, layer* previous = nullptr);

    void PopulateInput(const column& inputs);
    void ForwardsPass(const column& inputs);

    const int numNeurons;
    layer* previous;

    column activationValue;
    column gradients;
    matrix weights;
    double bias;
};

struct model
{
    layer* AddLayer(int numNeurons, layer* previousLayer = nullptr);

    void ForwardsPass(const column& inputs);
    double BackwardsPass(const column& targets, double learning_rate);
    void Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate);

    void PredictSingleInput(const column& inputs, column& outputs);

    std::vector<layer*> layers;

    double loss;
    int epoch = 0;
};
