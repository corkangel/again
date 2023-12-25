#include <iostream>

#include <vector>
#include <cassert>
#include <cmath>
#include <random>

typedef std::vector<double> column;
typedef std::vector<column> matrix;

// clamps the input between 0 and 1
double activation_function(const double input)
{
	// sigmoid function
	return 1  / (1 + exp(-input));
}

double activation_function_derivative(const double input)
{
	// derivative of sigmoid function
	const double df = activation_function(input);
	return df * (1 - df);
}

double cost_function(const double predicted, const double target)
{
    // MSE cost function - whose derivative below is a simple addition!
    return 0.5 * pow((predicted - target), 2);
}

double cost_function_derivative(const double predicted, const double target)
{
    return predicted - target;
}

// returns a random between -1 and 1
double random_value()
{
    return ((rand() / 2000) - 1000) * 0.001;
}

struct layer
{
    layer(int numNeurons, layer* previous = nullptr)
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

    const int numNeurons;
    layer* previous;

    column activationValue;
    column gradients;
    matrix weights;
    double bias;
};

struct model
{
    layer* AddLayer(int numNeurons, layer* previousLayer = nullptr)
    {
        layer* l = new layer(numNeurons, previousLayer);
        layers.push_back(l);
        return l;
    }

    std::vector<layer*> layers;
};

void populate_inputLayer(layer& layer, const column& inputs)
{
    assert(layer.numNeurons == inputs.size());
    for (int n=0; n < layer.numNeurons; n++)
    {
        layer.activationValue[n] = inputs[n];
    }    
    layer.bias = 1;
}

void layer_forward_pass(layer& layer, const column& inputs)
{
    assert(layer.weights[0].size() == inputs.size());

    for (int n=0; n < layer.numNeurons; n++)
    {
        double z = layer.bias;
        for (int i=0; i < inputs.size(); i++)
            z += layer.weights[n][i] * inputs[i];

        layer.activationValue[n] = activation_function(z);
    }
}

void model_forward_pass(model& m, const column& inputs)
{
    populate_inputLayer(*m.layers.front(), inputs);

    for (int l=1; l < m.layers.size(); l++)
        layer_forward_pass(*m.layers[l], m.layers[l-1]->activationValue);
}

double model_backwards_pass(model& m, const column& targets, double learning_rate)
{
    // the error in between the final layer and the targets
    double accumlatedError = 0;

    for (size_t l = m.layers.size()-1; l > 0; l--)
    {
        layer& currentLayer = *m.layers[l];

        const size_t numNeurons = currentLayer.activationValue.size();
        //column gradients(numNeurons);
        double cost = 0;

        for (int n=0; n < numNeurons; n++)
        {
            const double predicted = currentLayer.activationValue[n];

            if (l == m.layers.size()-1)
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
                layer* nextLayer = m.layers[l + 1];
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


void model_train(model& m, const matrix& allInputs, const matrix& allTargets, const int epochs)
{
    assert(allInputs.size() == allTargets.size());

    for (int e=0; e < epochs; e++)
    {
        double loss = 0;
        for (int i=0 ; i < allInputs.size(); i++)
        {
            model_forward_pass(m, allInputs[i]);
            loss += model_backwards_pass(m, allTargets[i], 0.1);
        }
        std::cout << "Epoch: " << e << " Loss: " << loss / allInputs.size() << " \n";
    }
}

int main(int, char**)
{
    // stable random values
    srand(101010101);

    const matrix inputs = {
        {0.1, 0.3},
        {.3, .6},
        {.2, .8}
    };

    const matrix targets = {
        {.4, .2, .1},
        {.5, .4, .2},
        {.7, .8, .6}
    };

    model m;
    layer* l = m.AddLayer(2); // input layer
    l = m.AddLayer(3, l); // hiddenA
    l = m.AddLayer(2, l); // hiddenB
    l = m.AddLayer(3,l); // output layer

    model_train(m, inputs, targets, 100);

}
