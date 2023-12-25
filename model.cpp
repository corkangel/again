#include <cassert>
#include <cmath>
#include <random>
#include <iostream>

#include "model.h"


// ------------------------------- utilities -------------------------------


// clamps the input between 0 and 1
double activation_function(const double input)
{
	// sigmoid function
	return 1  / (1 + exp(-input));

    // relu function
    //return std::max(0.0, input);
}

double activation_function_derivative(const double input)
{
	// derivative of sigmoid function
	const double df = activation_function(input);
	return df * (1 - df);

    // relu function
    //return (input > 0.0) ? 1.0 : 0.0;
}

double cost_function(const double predicted, const double target)
{
    // MSE cost function - whose derivative below is a simple addition!
    return 0.5 * pow((predicted - target), 2);

    // RMSE
    //return sqrt(0.5 * pow((predicted - target), 2));
}

double cost_function_derivative(const double predicted, const double target)
{
    // MSE
    return predicted - target;

    // RMSE
   // return (predicted - target) / sqrt(2.0);
}

// returns a random between 0 and 1
double random_value()
{
    return ((rand() / 1000)) * 0.001;
}

// ------------------------------- layer -------------------------------

layer::layer(int numNeurons, layer* previous)
    : numNeurons(numNeurons)
    , previous(previous)
{
    activationValue.resize(numNeurons);
    gradients.resize(numNeurons);

    if (previous)
    {
        weights.resize(numNeurons);
        for (auto&& w : weights)
            w.resize(previous->numNeurons);

        for (int i=0; i < numNeurons; i++)
            for (int j=0; j < previous->numNeurons; j++)
                weights[i][j] = random_value();
    }
    bias = random_value();
}


void layer::PopulateInput(const column& inputs)
{
    assert(numNeurons == inputs.size());
    for (int n=0; n < numNeurons; n++)
    {
        activationValue[n] = inputs[n];
    }    
    bias = 1;
}

void layer::ForwardsPass(const column& inputs)
{
    assert(weights[0].size() == inputs.size());

    for (int n=0; n < numNeurons; n++)
    {
        double z = bias;
        for (int i=0; i < inputs.size(); i++)
            z += weights[n][i] * inputs[i];

        activationValue[n] = activation_function(z);
    }
}

// ------------------------------- model -------------------------------

layer* model::AddLayer(int numNeurons, layer* previousLayer)
{
    layer* l = new layer(numNeurons, previousLayer);
    layers.push_back(l);
    return l;
}

void model::ForwardsPass(const column& inputs)
{
    layers.front()->PopulateInput(inputs);

    for (int l=1; l < layers.size(); l++)
        layers[l]->ForwardsPass(layers[l-1]->activationValue);
}

double model::BackwardsPass(const column& targets, double learning_rate)
{
    // the error in between the final layer and the targets
    double accumlatedError = 0;

    for (size_t l = layers.size()-1; l > 0; l--)
    {
        layer& currentLayer = *layers[l];

        const size_t numNeurons = currentLayer.activationValue.size();
        //column gradients(numNeurons);
        double cost = 0;

        for (int n=0; n < numNeurons; n++)
        {
            const double predicted = currentLayer.activationValue[n];

            if (l == layers.size()-1)
            {
                // output layer
                const double target = targets[n];
                cost = cost_function_derivative(predicted, target);

                // this is just for reporting - not used in the calculations
                accumlatedError += cost_function(predicted, target);
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

                    cost += connectionWeight * errorTerm;
                }
            }

            // Compute the delta/gradient
            currentLayer.gradients[n] = cost * activation_function_derivative(predicted);

            // Update weights
            for (int i = 0; i < currentLayer.previous->numNeurons; ++i)
            {
                // the input is the activation value of the neuron in the previous layer
                const double input = currentLayer.previous->activationValue[i];

                currentLayer.weights[n][i] -= learning_rate * currentLayer.gradients[n] * input;
            }

            // Update bias
            currentLayer.bias -= learning_rate * currentLayer.gradients[n]; // bias input is always 1, so is omitted
        }
    }
    return accumlatedError;
}

void model::Train(const matrix& allInputs, const matrix& allTargets, const int epochs)
{
    assert(allInputs.size() == allTargets.size());

    for (int e=0; e < epochs; e++)
    {
        loss = 0;
        for (int i=0 ; i < allInputs.size(); i++)
        {
            ForwardsPass(allInputs[i]);
            loss += BackwardsPass(allTargets[i], 0.1);
        }
        // if (e%20 == 0)
        //     std::cout << "Epoch: " << e << " Loss: " << loss / allInputs.size() << " \n";
    }
}
