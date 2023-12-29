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
    CrossEntropy,
    Last
};
using CostFuncPtr = double (*)(const double, const double);

struct layer
{
    layer(uint32 numNeurons);
    ~layer() {}

    virtual void ForwardsPass(const column& inputs);

    double BackwardsPass(
        const layer& previousLayer,
        const layer* nextLayer,
        const double learning_rate,
        const column& targets,
        CostFuncPtr cf,
        CostFuncPtr cfD);

    const uint32 numNeurons;

    column activationValue;
    column gradients;
    column errors;
    matrix weights;
    double bias;
    bool forClassification;

    ActivationFuncPtr af;
    ActivationFuncPtr afD;


};

struct denseLayer : layer
{
    denseLayer(
        uint32 numNeurons, 
        ActivationFunction aFunc,
        layer* previous = nullptr);

    void ForwardsPass(const column& inputs) override;

    ActivationFunction aFunc;
};

struct model
{
    model();

    layer* AddInputLayer(uint32 numNeurons);

    layer* AddDenseLayer(
        uint32 numNeurons, 
        ActivationFunction aFunc, 
        layer* previousLayer);

    void ForwardsPass(const column& inputs);
    double BackwardsPass(const column& targets, double learning_rate);
    void Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate);

    void PredictSingleInput(const column& inputs, column& outputs);

    std::vector<layer*> layers;

    CostFunction cFunc;
    CostFuncPtr cf;
    CostFuncPtr cfD;

    double loss;
    int epoch = 0;
};
