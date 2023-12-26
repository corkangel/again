#include <cassert>
#include <cmath>
#include <random>
#include <iostream>

#include "model.h"

// ------------------------------- activation functons -------------------------------

double activation_function_sigmoid(const double input)
{
	return 1  / (1 + exp(-input));
}

double activation_function_sigmoid_derivative(const double input)
{
	const double df = activation_function_sigmoid(input);
	return df * (1 - df);
}

double activation_function_relu(const double input)
{
    return std::max(0.0, input);
}

double activation_function_relu_derivative(const double input)
{
    return (input > 0.0) ? 1.0 : 0.0;
}

static ActivationFuncPtr activationFuncPtrs[4][2] = {
    {nullptr, nullptr},
    {activation_function_sigmoid, activation_function_sigmoid_derivative},
    {activation_function_relu, activation_function_relu_derivative},
    {nullptr, nullptr}
};

// ------------------------------- cost functions -------------------------------

double cost_function_mse(const double predicted, const double target)
{
    // MSE cost function - whose derivative below is a simple addition!
    return 0.5 * pow((predicted - target), 2);
}

double cost_function_mse_derivative(const double predicted, const double target)
{
    // MSE
    return predicted - target;
}

double cost_function_rmse(const double predicted, const double target)
{
    // RMSE
    return sqrt(0.5 * pow((predicted - target), 2));
}

double cost_function_rmse_derivative(const double predicted, const double target)
{
    // RMSE
    return (predicted - target) / sqrt(2.0);
}

static CostFuncPtr costFuncPtrs[4][2] = {
    {nullptr, nullptr},
    {cost_function_mse, cost_function_mse_derivative},
    {cost_function_rmse, cost_function_rmse_derivative},
    {nullptr, nullptr}
};

// ------------------------------- utils -------------------------------

// returns a random between 0 and 1
double random_value()
{
    return ((rand() / 1000)) * 0.001;
}

// ------------------------------- layer -------------------------------

layer::layer(int numNeurons)
    : numNeurons(numNeurons)
{
    activationValue.resize(numNeurons);
    gradients.resize(numNeurons);
}

void layer::ForwardsPass(const column& inputs)
{
    // blindly copy the input values
    assert(numNeurons == inputs.size());
    for (int n=0; n < numNeurons; n++)
    {
        activationValue[n] = inputs[n];
    }    
    bias = 1;
}

// ------------------------------- denseLayer -------------------------------

denseLayer::denseLayer(int numNeurons, ActivationFunction aFunc, CostFunction cFunc,  layer* previous)
    : layer(numNeurons)
    , aFunc(aFunc)
    , cFunc(cFunc)
{
    assert(previous);

    weights.resize(numNeurons);
    for (auto&& w : weights)
        w.resize(previous->numNeurons);

    for (int i=0; i < numNeurons; i++)
        for (int j=0; j < previous->numNeurons; j++)
            weights[i][j] = random_value();

    bias = random_value();

    af = activationFuncPtrs[int(aFunc)][0];
    afD = activationFuncPtrs[int(aFunc)][1];

    cf = costFuncPtrs[int(cFunc)][0];
    cfD = costFuncPtrs[int(cFunc)][1];
}

void denseLayer::ForwardsPass(const column& inputs)
{
    assert(weights[0].size() == inputs.size());

    for (int n=0; n < numNeurons; n++)
    {
        double z = bias;
        for (int i=0; i < inputs.size(); i++)
            z += weights[n][i] * inputs[i];

        activationValue[n] = af(z);
    }
}

// ------------------------------- model -------------------------------

layer* model::AddInputLayer(int numNeurons)
{
    layer* l = new layer(numNeurons);
    layers.push_back(l);
    return l;
}

layer* model::AddDenseLayer(
    int numNeurons, 
    ActivationFunction aFunc,
    CostFunction cFunc,
    layer* previousLayer)
{
    layer* l = new denseLayer(numNeurons, aFunc, cFunc, previousLayer);
    layers.push_back(l);
    return l;
}

void model::ForwardsPass(const column& inputs)
{
    layers.front()->ForwardsPass(inputs);

    for (int l=1; l < layers.size(); l++)
        layers[l]->ForwardsPass(layers[l-1]->activationValue);
}

void model::PredictSingleInput(const column& inputs, column& outputs)
{
    assert(inputs.size() == layers.front()->numNeurons);
    assert(outputs.size() == layers.back()->numNeurons);

    layers.front()->ForwardsPass(inputs);

    for (int l=1; l < layers.size(); l++)
        layers[l]->ForwardsPass(layers[l-1]->activationValue);

    const layer& outputLayer = *layers.back();
    for (int i=0; i < outputLayer.numNeurons; i++)
        outputs[i] = outputLayer.activationValue[i];
}

double model::BackwardsPass(const column& targets, double learning_rate)
{
    // the error in between the final layer and the targets
    double accumlatedError = 0;

    for (size_t l = layers.size()-1; l > 0; l--)
    {
        layer& currentLayer = *layers[l];
        layer& previousLayer = *layers[l-1];

        const size_t numNeurons = currentLayer.numNeurons;
        double cost = 0;

        for (int n=0; n < numNeurons; n++)
        {
            const double predicted = currentLayer.activationValue[n];

            if (l == layers.size()-1)
            {
                // output layer
                const double target = targets[n];
                cost = currentLayer.cfD(predicted, target);

                // this is just for reporting - not used in the calculations
                accumlatedError += pow(currentLayer.cf(predicted, target),2);
            }
            else
            {
                // hidden layers
                layer* nextLayer = layers[l + 1];
                for (int k = 0; k < nextLayer->numNeurons; ++k)
                {
                    // the error term associated with a neuron in the next layer
                    const double errorTerm = nextLayer->gradients[k];

                    // the weight connecting neuron k in the next layer to neuron n in the current layer
                    const double connectionWeight = nextLayer->weights[k][n];

                    //NOTE: NOT cost += here!?
                    cost = connectionWeight * errorTerm;
                }
            }

            // Compute the delta/gradient
            currentLayer.gradients[n] = cost * currentLayer.afD(predicted);

            // Update weights
            for (int i = 0; i < previousLayer.numNeurons; ++i)
            {
                // the input is the activation value of the neuron in the previous layer
                const double input = previousLayer.activationValue[i];

                currentLayer.weights[n][i] -= learning_rate * currentLayer.gradients[n] * input;
            }

            // Update bias
            currentLayer.bias -= learning_rate * currentLayer.gradients[n]; // bias input is always 1, so is omitted
        }
    }
    return accumlatedError;
}

void model::Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate)
{
    assert(allInputs.size() == allTargets.size());

    for (int e=0; e < epochs; e++)
    {
        loss = 0;
        for (int i=0 ; i < allInputs.size(); i++)
        {
            ForwardsPass(allInputs[i]);
            loss += BackwardsPass(allTargets[i], learningRate);
        }
        loss = loss / allInputs.size() ;

         if (e%20 == 0)
             std::cout << "Epoch: " << e << " Loss: " << loss  << " \n";
    }
    epoch += epochs;
}
