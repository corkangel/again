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

double activation_function_softmax(const double input)
{
    // just return the input, it'll be classified later
    return input;
}

double activation_function_softwmax_derivative(const double input)
{
    // ??? TODO
    return input;
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
    {activation_function_softmax, activation_function_softwmax_derivative},
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
    // un-used, since softmax is run over the entire array
    assert(0);
    return 0;
}

double cost_function_crossEntropy_derivative(const double predicted, const double target)
{
    // simplifies to simple addition!
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
    return ((rand() / 1000)) * 0.001;
}

// ------------------------------- layer -------------------------------

layer::layer(uint32 numNeurons)
    : numNeurons(numNeurons)
    , forClassification(false)
{
    activationValue.resize(numNeurons);
    gradients.resize(numNeurons);
}

void layer::ForwardsPass(const column& inputs)
{
    // blindly copy the input values
    assert(numNeurons == inputs.size());
    for (uint32 n=0; n < numNeurons; n++)
    {
        activationValue[n] = inputs[n];
    }    
    bias = 1;
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

    for (uint32 i=0; i < numNeurons; i++)
        for (uint32 j=0; j < previous->numNeurons; j++)
            weights[i][j] = random_value();

    bias = random_value();

    af = activationFuncPtrs[int(aFunc)][0];
    afD = activationFuncPtrs[int(aFunc)][1];

}

void denseLayer::ForwardsPass(const column& inputs)
{
    assert(weights[0].size() == inputs.size());

    for (uint32 n=0; n < numNeurons; n++)
    {
        double z = bias;
        for (int i=0; i < inputs.size(); i++)
            z += weights[n][i] * inputs[i];

        activationValue[n] = af(z);
        //assert(!isnan(activationValue[n]) && !isinf(activationValue[n]) && activationValue[n] < 10000);
        if ( activationValue[n] > 10000)
        {
            int d = 1;
            d++;
        }
    }

    if (forClassification)
        activationValue = softmax(activationValue);
}


double layer::BackwardsPass(
    const layer& previousLayer,
    const layer* nextLayer,
    const double learning_rate,
    const column& targets,
    CostFuncPtr cf,
    CostFuncPtr cfD)
{
    double err = 0;
    double derivativeOfCost = 0;

    for (uint32 n=0; n < numNeurons; n++)
    {
        const double predicted = activationValue[n];
        
        if (nextLayer == nullptr)
        {
            // this is the output layer
            assert(targets.size() == numNeurons);
            const double target = targets[n];

            err += cf(predicted, target);
            derivativeOfCost = cfD(predicted, target);
            assert(!isnan(derivativeOfCost) && !isinf(derivativeOfCost));

            // this is just for reporting - not used in the calculations
            //accumlatedError += pow(cfD(predicted, target),2);
        }
        else
        {
            // hidden layers
            for (uint32 k = 0; k < nextLayer->numNeurons; ++k)
            {
                // the error term associated with a neuron in the next layer
                const double errorTerm = nextLayer->gradients[k];

                // the weight connecting neuron k in the next layer to neuron n in the current layer
                const double connectionWeight = nextLayer->weights[k][n];

                //NOTE: NOT cost += here!?
                derivativeOfCost = connectionWeight * errorTerm;
                assert(!isnan(derivativeOfCost) && !isinf(derivativeOfCost));
            }
        }

        // Compute the delta/gradient
        gradients[n] = derivativeOfCost ;//* afD(predicted);

        // Update weights
        for (uint32 i = 0; i < previousLayer.numNeurons; ++i)
        {
            // the input is the activation value of the neuron in the previous layer
            const double input = previousLayer.activationValue[i];

            weights[n][i] -= learning_rate * gradients[n] * input;
            assert(!isnan(weights[n][i]) && !isinf(weights[n][i]) && weights[n][i] < 10000);

            // if (l==1 && n == 0 && i == 0)
            // {
            //     printf("weight zero: p:%.4f w:%.4f g:%.4f\n", 
            //             activationValue[n],
            //         weights[n][i], 
            //         gradients[n]);
            // }
        }

        // Update bias
        bias -= learning_rate * gradients[n]; // bias input is always 1, so is omitted
    }
    return err;
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
    // the error in between the final layer and the targets
    double accumlatedError = 0;

    layer* outputLayer = layers.back();
    accumlatedError += outputLayer->BackwardsPass(*layers[layers.size()-2], nullptr, learning_rate, targets, cf, cfD);

    // other layers
    for (uint32 l =uint32(layers.size()-2); l > 0; l--)
    {
        column targets; // un-used
        layer& currentLayer = *layers[l];
        layer& previousLayer = *layers[l-1];
        layer* nextLayer = layers[l+1];
        accumlatedError += currentLayer.BackwardsPass(previousLayer, nextLayer, learning_rate, targets, cf, cfD);
    }
    return accumlatedError;
}

void model::Train(const matrix& allInputs, const matrix& allTargets, const int epochs, const double learningRate)
{
    assert(allInputs.size() == allTargets.size());

    for (int e=0; e < epochs; e++)
    {
        // run just one of the inputs
        int I = rand() % allInputs.size();
        ForwardsPass(allInputs[I]);
        loss = BackwardsPass(allTargets[I], learningRate);

    }
    epoch += epochs;
}
