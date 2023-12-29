#include <cassert>

#include "model.h"


bool nothing()
{
    model m;
    return true;
}

bool layers()
{
    {
        // no input layers
        model m;
        assert(m.AddDenseLayer(1, ActivationFunction::Sigmoid, nullptr) == nullptr); 
    }

    {
        // multiple input layers
        model m;
        m.AddInputLayer(2);
        m.AddInputLayer(2);
    }

    {
        model m;
        layer* l = m.AddInputLayer(2);
        l = m.AddDenseLayer(1, ActivationFunction::Sigmoid, l);

        // negative numNeurons
        assert(m.AddDenseLayer(-1, ActivationFunction::Relu, l) == nullptr);

        // null previousLayer
        assert(m.AddDenseLayer(1, ActivationFunction::Relu, nullptr) == nullptr);
    }

    return true;
}

const matrix simpleInputs = {
    {.1, .1},
    {.5, .5}
};

const matrix simpleTargets = { {1}, {5} };

void initSimpleModel(model& m, ActivationFunction af)
{
    layer* l = m.AddInputLayer(2);
    l = m.AddDenseLayer(2, af, l);
    layer* outputLayer = m.AddDenseLayer(1, af, l);
}

const uint32 softmaxTestSize = 6;
const column softmaxTargets = { {.2, 0.5, 0, 0, 0.1, .3} };


bool predict()
{
    // run the prediction code without any training.

    {
        model m;
        initSimpleModel(m, ActivationFunction::Sigmoid);

        column out1(1), out2(1);
        m.PredictSingleInput(simpleInputs[0], out1);
        m.PredictSingleInput(simpleInputs[0], out2);
        assert(out1[0] == out2[0]);
    }

    {
        model m;
        initSimpleModel(m, ActivationFunction::Relu);

        column out1(1), out2(1);
        m.PredictSingleInput(simpleInputs[0], out1);
        m.PredictSingleInput(simpleInputs[0], out2);
        assert(out1[0] == out2[0]);
    }

    {
        model m;
        layer* l = m.AddInputLayer(2);
        l = m.AddDenseLayer(2, ActivationFunction::Relu, l);
        layer* outputLayer = m.AddDenseLayer(softmaxTestSize, ActivationFunction::Softmax, l);

        column out1(softmaxTestSize), out2(softmaxTestSize);
        m.PredictSingleInput(simpleInputs[0], out1);
        m.PredictSingleInput(simpleInputs[0], out2);
        assert(out1[0] == out2[0]);
        assert(argmax(out1) == argmax(out2));

        double total = 0;
        for (auto o : out1) total+= o;
        assert(total == 1);
    }

    
    // test individual neuron activation function value
    {
        layer il(2);
        denseLayer dl(1, ActivationFunction::Sigmoid, &il);

        dl.biases = {0.3};
        dl.weights[0][0] = 0.23;
        dl.weights[0][1] = -0.1;

        column inputs(2);
        inputs[0] = 1;
        inputs[1] = 0;

        dl.ForwardsPass(inputs);

        const double expectedActivationValue = 0.6295;
        assert(abs(dl.activationValue[0]-expectedActivationValue) < 0.001);
    }

    {
        // just to get cfD
        model m;

        layer il(1);
        denseLayer dl(1, ActivationFunction::Sigmoid, &il);
        denseLayer ol(1, ActivationFunction::Sigmoid, &dl);        

        dl.biases = {0.3};
        dl.weights[0][0] = 0.3;

        column inputs(1);
        inputs[0] = 1;

        column targets(1);
        targets[0] = 0;

        dl.ForwardsPass(inputs);
        
        ol.BackwardsPass(dl, nullptr, 0.5, targets, m.cf, m.cfD);
        dl.BackwardsPass(il, &ol, 0.5, ol.activationValue, m.cf, m.cfD);

    }


    return true;
}

bool backwards()
{
    {
        model m;
        initSimpleModel(m, ActivationFunction::Sigmoid);

        column out1(1);

        m.PredictSingleInput(simpleInputs[0], out1);
        double loss1 = m.BackwardsPass(simpleTargets[0], 0.1);

        m.PredictSingleInput(simpleInputs[1], out1);
        double loss2 = m.BackwardsPass(simpleTargets[0], 0.1);

        assert(loss2 != loss1);
    }

    {
        model m;
        initSimpleModel(m, ActivationFunction::Relu);

        column out1(1);

        m.PredictSingleInput(simpleInputs[0], out1);
        double loss1 = m.BackwardsPass(simpleTargets[0], 0.1);

        m.PredictSingleInput(simpleInputs[1], out1);
        double loss2 = m.BackwardsPass(simpleTargets[0], 0.1);

        assert(loss2 != loss1);
    }

    {
        model m;
        layer* l = m.AddInputLayer(2);
        l = m.AddDenseLayer(2, ActivationFunction::Relu, l);
        layer* outputLayer = m.AddDenseLayer(softmaxTestSize, ActivationFunction::Softmax, l);

        column out1(softmaxTestSize);

        m.PredictSingleInput(simpleInputs[0], out1);
        double loss1 = m.BackwardsPass(softmaxTargets, 0.1);

        m.PredictSingleInput(simpleInputs[1], out1);
        double loss2 = m.BackwardsPass(softmaxTargets, 0.1);

        assert(loss2 != loss1);
    }

    return true;
}

// numbers training set.
// doubles and their corresponding integers

const uint32 numbersBatchSize = 20;
matrix numbersBatchDoubles;
matrix numbersBatchIntegers;
matrix numbersTestDoubles;
matrix numbersTestIntegers;

void initNumbers()
{
    srand(10910910);
    numbersBatchDoubles.resize(numbersBatchSize);
    numbersBatchIntegers.resize(numbersBatchSize);
    numbersTestDoubles.resize(numbersBatchSize);
    numbersTestIntegers.resize(numbersBatchSize);

    for (uint32 i=0; i < numbersBatchSize; i++)
    {
        numbersBatchDoubles[i].resize(1);
        numbersBatchDoubles[i][0] = (rand() % 1000) * 0.01;

        numbersBatchIntegers[i].resize(1);
        numbersBatchIntegers[i][0] = (double)(uint32)numbersBatchDoubles[i][0];

        numbersTestDoubles[i].resize(1);
        numbersTestDoubles[i][0] = (rand() % 1000) * 0.01;

        numbersTestIntegers[i].resize(1);
        numbersTestIntegers[i][0] = (double)(uint32)numbersTestDoubles[i][0];
    }
}


double trainNumbers(model& m, uint32 numEpochs)
{
    m.Train(numbersBatchDoubles, numbersBatchIntegers, numEpochs, 0.1);

    //printf("trainNumbers loss: %f\n", m.loss);

    double loss = 0;
    column predictions(1);
    for (uint32 i=0; i < numbersBatchSize; i++)
    {
        m.PredictSingleInput(numbersTestDoubles[i], predictions);

        loss += pow(predictions[0] - numbersTestIntegers[i][0], 2);
    }

    return loss / numbersBatchSize;
}

bool train()
{
    initNumbers();

    model m;

    layer* l = m.AddInputLayer(1);
    l = m.AddDenseLayer(8, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(1, ActivationFunction::Relu, l);

    for (uint32 i=0; i <20; i++)
    {
        double loss = trainNumbers(m, 10);
        printf("train numbers loss: %f\n", loss);
    }
    return true;
}


const matrix seedsDataset =
{
    {2.7810836,2.550537003},
    {1.465489372,2.362125076},
    {3.396561688,4.400293529},
    {1.38807019,1.850220317},
    {3.06407232,3.005305973},
    {7.627531214,2.759262235},
    {5.332441248,2.088626775},
    {6.922596716,1.77106367},
    {8.675418651,-0.242068655},
    {7.673756466,3.508563011}
};

// all the outputs are the same?!
const matrix seedsOutputs = { 
    {1,0},
    {1,0},
    {1,0},
    {1,0},
    {1,0},
    {0,1},
    {0,1},
    {0,1},
    {0,1},
    {0,1}
};


bool seeds()
{
    model m;
    layer* l = m.AddInputLayer(2);
    layer* hidden = m.AddDenseLayer(2, ActivationFunction::Sigmoid, l);
    layer* output = m.AddDenseLayer(2, ActivationFunction::Sigmoid, hidden);

    hidden->weights = { {0.13436424411240122, 0.8474337369372327}, {0.2550690257394217,  0.49543508709194095}};
    hidden->biases = {0.763774618976614, 0.4494910647887381};

    output->weights = { {0.651592972722763, 0.7887233511355132}, {0.02834747652200631, 0.8357651039198697}};
    output->biases = {0.0938595867742349, 0.43276706790505337};

    for (uint32 i = 0; i < 20; i++)
    {
        m.Train(seedsDataset, seedsOutputs, 1, 0.5);
        printf("seeds -  loss sum:%f act:%f grad:%f err:%f\n", m.loss, output->activationValue[0],output->gradients[0],  output->errors[0]);
    }

    return true;
}

void check(const char* name, const int result)
{
    printf("TEST: [%-12s] %s\n", name, result ? "success" : "fail");
}

int main(int, char**)
{
    printf("tests begin\n");
    check("nothing", nothing());
    check("layers", layers());
    check("predict", predict());
    check("backwards", backwards());
    check("train", train());
    check("seeds", seeds());
    printf("tests end\n");
    return 1;
}