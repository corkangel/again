#include <cassert>
#include <cmath>
#include <random>
#include <iostream>

#include "model.h"

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

// ------------------------------- activation functons -------------------------------

double activation_function_sigmoid(const double input)
{
	return 1  / (1 + exp(-input));
}

double activation_function_sigmoid_derivative(const double input)
{
	return input * (1 - input);
}

double activation_function_relu(const double input)
{
    return std::max(0.0, input);
}

double activation_function_relu_derivative(const double input)
{
    return (input > 0.0) ? 1.0 : 0.0;
}

double activation_function_softmax(const double input)
{
    // just return the input, it'll be classified later
    return input;
}

double activation_function_softmax_derivative(const double input)
{
    return input * (1 - input);
}

// takes an array of all the inputs, not just a single input
column softmax(const column& inputs) 
{
    double maxInput = *std::max_element(inputs.begin(), inputs.end());
    column expValues(inputs.size());
    double sumExpValues = 0.0;

    // Compute the exponentials and sum them
    for (size_t i = 0; i < inputs.size(); ++i) 
    {
        expValues[i] = exp(inputs[i] - maxInput);  // Subtract maxInput for numerical stability
        sumExpValues += expValues[i];
    }

    // Normalize to get probabilities
    for (double& val : expValues) {
        assert(!isnan(val));
        val /= sumExpValues;
    }

    return expValues;
}

static ActivationFuncPtr activationFuncPtrs[5][2] = {
    {nullptr, nullptr},
    {activation_function_sigmoid, activation_function_sigmoid_derivative},
    {activation_function_relu, activation_function_relu_derivative},
    {activation_function_softmax, activation_function_softmax_derivative},
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
    return sqrt(0.5 * pow((predicted - target), 2));
}

double cost_function_rmse_derivative(const double predicted, const double target)
{
    return (predicted - target) / sqrt(2.0);
}

double cost_function_crossEntropy(const double predicted, const double target)
{
    // un-used, since softmax is run over the entire array. 
    return - target*exp(predicted);
}

double cost_function_crossEntropy_derivative(const double predicted, const double target) 
{
    return predicted - target;
}

static CostFuncPtr costFuncPtrs[5][2] = {
    {nullptr, nullptr},
    {cost_function_mse, cost_function_mse_derivative},
    {cost_function_rmse, cost_function_rmse_derivative},
    {cost_function_crossEntropy, cost_function_crossEntropy_derivative},
    {nullptr, nullptr}
};

// ------------------------------- utils -------------------------------

// returns a random between 0 and 1
double random_value()
{
    return ((rand() % 1000)) * 0.001;
}

// ------------------------------- layer -------------------------------

layer::layer(uint32 numNeurons)
    : numNeurons(numNeurons)
    , forClassification(false)
{
    activationValue.resize(numNeurons);
    gradients.resize(numNeurons);
    errors.resize(numNeurons);
    biases.resize(numNeurons);
}

void layer::ForwardsPass(const column& inputs)
{
    // blindly copy the input values
    assert(numNeurons == inputs.size());
    for (uint32 n=0; n < numNeurons; n++)
    {
        activationValue[n] = inputs[n];
        biases[n] = 1;
    }    
}

// ------------------------------- denseLayer -------------------------------

denseLayer::denseLayer(uint32 numNeurons, ActivationFunction aFunc, layer* previous)
    : layer(numNeurons)
    , aFunc(aFunc)
{
    assert(previous);

    weights.resize(numNeurons);
    for (auto&& w : weights)
        w.resize(previous->numNeurons);

    for (uint32 n=0; n < numNeurons; n++)
        for (uint32 j=0; j < previous->numNeurons; j++)
            weights[n][j] = random_value();

    for (uint32 n=0; n < numNeurons; n++)
        biases[n] = random_value();

    af = activationFuncPtrs[int(aFunc)][0];
    afD = activationFuncPtrs[int(aFunc)][1];

}

void denseLayer::ForwardsPass(const column& inputs)
{
    assert(weights[0].size() == inputs.size());

    for (uint32 n=0; n < numNeurons; n++)
    {
        double z = biases[n];
        for (int i=0; i < inputs.size(); i++)
            z += weights[n][i] * inputs[i];

        activationValue[n] = af(z);
        assert(!isnan(activationValue[n]) && !isinf(activationValue[n]));// && activationValue[n] < 100000);
        //printf("ACT: %f/%f  i:%f,%f  w:%f,%f,%f,%f\n", z, activationValue[n], inputs[0], inputs[1], 
        //    weights[0][0], weights[0][1], weights[1][0], weights[1][1]);
    }

    if (forClassification)
        activationValue = softmax(activationValue);
}

double dotProduct(const column& a, const column& b)
{
    assert(a.size() == b.size());
    double sum = 0;
    for (int i=0; i < a.size(); i++)
        sum += a[i] * b[i];
    return sum;
}

double layer::BackwardsPass(
    const layer& previousLayer,
    const layer* nextLayer,
    const double learning_rate,
    const column& targets,
    CostFuncPtr cf,
    CostFuncPtr cfD)
{
    double accumulatedError = 0;
    for (uint32 n=0; n < numNeurons; n++)
    {
        const double predicted = activationValue[n];
        errors[n] = 0;
        
        if (nextLayer == nullptr)
        {
            // this is the output layer
            errors[n] = cfD(predicted, targets[n]);

            // only for reporting
            accumulatedError += pow(cf(predicted, targets[n]),2);
        }
        else
        {
            for (uint32 k=0; k < nextLayer->numNeurons; k++)
            {
                errors[n] += nextLayer->weights[k][n] * nextLayer->gradients[k];
            }
        }

        if (!forClassification)
        {
            gradients[n] = errors[n] * afD(predicted);
        }
    }
     
    if (forClassification)
    {
        for (uint32 n=0; n < numNeurons; n++)
        {
             //const double predicted = activationValue[n];
             //gradients[n] = predicted - targets[n];

            const double predicted = activationValue[n];
            column tmp(numNeurons);
            for (uint32 j=0; j < numNeurons; j++)
            {
                if (j == n)
                    tmp[j] = predicted * (1 - predicted);
                else
                    tmp[j] = -predicted * activationValue[j];
            }
            gradients[n] = dotProduct(errors, tmp);
        }
    }


    // Update weights
    for (uint32 n=0; n < numNeurons; n++)
    {
        for (uint32 i = 0; i < previousLayer.numNeurons; ++i)
        {
            // the input is the activation value of the neuron in the previous layer
            const double input = previousLayer.activationValue[i];
            weights[n][i] -= learning_rate * gradients[n] * input;
        }

        // Update bias
        biases[n] -= learning_rate * gradients[n]; // bias input is always 1, so is omitted
    }
    return pow(accumulatedError,2);
}

// ------------------------------- model -------------------------------

const int MaxNeurons = 1000 * 1000;

model::model()

{
    srand(999999);

    cFunc = CostFunction::MSE;    
    cf = costFuncPtrs[int(cFunc)][0];
    cfD = costFuncPtrs[int(cFunc)][1];
}

layer* model::AddInputLayer(uint32 numNeurons)
{
    if (!layers.empty() || numNeurons > MaxNeurons)
    {
        return nullptr;
    }

    layer* l = new layer(numNeurons);
    layers.push_back(l);
    return l;
}

layer* model::AddDenseLayer(
    uint32 numNeurons, 
    ActivationFunction aFunc,
    layer* previousLayer)
{
    if (layers.empty() || numNeurons > MaxNeurons || previousLayer == nullptr)
    {
        return nullptr;
    }

    layer* l = new denseLayer(numNeurons, aFunc, previousLayer);
    layers.push_back(l);

    if (aFunc == ActivationFunction::Softmax)
    {
        l->forClassification = true;

        cFunc = CostFunction::CrossEntropy;    
        cf = costFuncPtrs[int(cFunc)][0];
        cfD = costFuncPtrs[int(cFunc)][1];
    }

    return l;
}

void model::ForwardsPass(const column& inputs)
{
    layers.front()->ForwardsPass(inputs);

    for (int l=1; l < layers.size(); l++)
        layers[l]->ForwardsPass(layers[l-1]->activationValue);

    //for (double a : layers.back()->activationValue)
    //    printf("a: %f ", a);

    //printf("\n");
}

void model::PredictSingleInput(const column& inputs, column& outputs)
{
    assert(inputs.size() == layers.front()->numNeurons);
    assert(outputs.size() == layers.back()->numNeurons);

    layers.front()->ForwardsPass(inputs);

    for (uint32 l=1; l < layers.size(); l++)
        layers[l]->ForwardsPass(layers[l-1]->activationValue);

    const layer& outputLayer = *layers.back();
    for (uint32 i=0; i < outputLayer.numNeurons; i++)
        outputs[i] = outputLayer.activationValue[i];
}

double model::BackwardsPass(const column& targets, double learning_rate)
{
    layer* outputLayer = layers.back();
    double accumlatedError = outputLayer->BackwardsPass(*layers[layers.size()-2], nullptr, learning_rate, targets, cf, cfD);

    // other layers
    for (uint32 l =uint32(layers.size()-2); l > 0; l--)
    {
        column dummytargets; // un-used
        layer& currentLayer = *layers[l];
        layer& previousLayer = *layers[l-1];
        layer* nextLayer = layers[l+1];
        currentLayer.BackwardsPass(previousLayer, nextLayer, learning_rate, dummytargets, cf, cfD);
    }
    return accumlatedError;
}

void model::Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate)
{
    assert(allInputs.size() == allTargets.size());

    for (int e=0; e < epochs; e++)
    {
        loss = 0;
        const size_t sz = allInputs.size();
        for (size_t i = 0; i < sz; i++)
        {
            ForwardsPass(allInputs[i]);
            loss += BackwardsPass(allTargets[i], learningRate);
        }
        // run just one of the inputs
        // int I = rand() % allInputs.size();
        // ForwardsPass(allInputs[I]);
        // loss = BackwardsPass(allTargets[I], learningRate);

    }
    epoch += epochs;
}
