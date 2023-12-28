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

    // {
    //     model m;
    //     layer* l = m.AddInputLayer(2);
    //     l = m.AddDenseLayer(2, ActivationFunction::Relu, l);
    //     layer* outputLayer = m.AddDenseLayer(softmaxTestSize, ActivationFunction::Softmax, l);

    //     column out1(softmaxTestSize), out2(softmaxTestSize);
    //     m.PredictSingleInput(simpleInputs[0], out1);
    //     m.PredictSingleInput(simpleInputs[0], out2);
    //     assert(out1[0] == out2[0]);
    //     assert(argmax(out1) == argmax(out2));

    //     double total = 0;
    //     for (auto o : out1) total+= o;
    //     assert(total == 1);
    // }

    
    // test individual neuron activation function value
    {
        layer il(2);
        denseLayer dl(1, ActivationFunction::Sigmoid, &il);

        dl.bias = 0.3;
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

        dl.bias = 0.3;
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

    // {
    //     model m;
    //     layer* l = m.AddInputLayer(2);
    //     l = m.AddDenseLayer(2, ActivationFunction::Relu, l);
    //     layer* outputLayer = m.AddDenseLayer(softmaxTestSize, ActivationFunction::Softmax, l);

    //     column out1(softmaxTestSize);

    //     m.PredictSingleInput(simpleInputs[0], out1);
    //     double loss1 = m.BackwardsPass(softmaxTargets, 0.1);

    //     m.PredictSingleInput(simpleInputs[1], out1);
    //     double loss2 = m.BackwardsPass(softmaxTargets, 0.1);

    //     assert(loss2 != loss1);
    // }

    return true;
}

// numbers training set.
// doubles and their corresponding integers

const uint32 numbersBatchSize = 100;
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
    l = m.AddDenseLayer(2, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(1, ActivationFunction::Sigmoid, l);

    for (uint32 i=0; i < 10; i++)
    {
        double loss = trainNumbers(m, 1000);
        printf("train numbers loss: %f\n", loss);
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
    printf("tests end\n");
    return 1;
}