#pragma once

#include "utils.h"

enum class ActivationFunction : short
{
    None,
    Sigmoid,
    Relu,
    Softmax,
    Last
};
using ActivationFuncPtr = double (*)(const double);

enum class CostFunction : short
{
    None,
    MSE,
    RMSE,
    Last
};
using CostFuncPtr = double (*)(const double, const double);

struct layer
{
    layer(int numNeurons);
    ~layer() {}

    virtual void ForwardsPass(const column& inputs);

    const int numNeurons;

    column activationValue;
    column gradients;
    matrix weights;    
    double bias;

    ActivationFuncPtr af;
    ActivationFuncPtr afD;

    CostFuncPtr cf;
    CostFuncPtr cfD;
};

struct denseLayer : layer
{
    denseLayer(
        int numNeurons, 
        ActivationFunction aFunc,
        CostFunction cFunc,
        layer* previous = nullptr);

    void ForwardsPass(const column& inputs) override;

    ActivationFunction aFunc;
    CostFunction cFunc;
};

struct model
{
    layer* AddInputLayer(int numNeurons);

    layer* AddDenseLayer(
        int numNeurons, 
        ActivationFunction aFunc, 
        CostFunction cFunc,
        layer* previousLayer);

    void ForwardsPass(const column& inputs);
    double BackwardsPass(const column& targets, double learning_rate);
    void Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate);

    void PredictSingleInput(const column& inputs, column& outputs);

    std::vector<layer*> layers;

    double loss;
    int epoch = 0;
};
