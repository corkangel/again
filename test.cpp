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

// numbers training set

const uint32 numbersBatchSize = 100;
matrix numbersBatchDoubles;
matrix numbersBatchIntegers;
matrix numbersTestDoubles;
matrix numbersTestIntegers;

void initNumbers()
{
    srand(10910910);
    numbersBatchDoubles.resize(numbersBatchSize);
    for (auto&& b : numbersBatchDoubles) b.resize(1);

    numbersBatchIntegers.resize(numbersBatchSize);
    for (auto&& b : numbersBatchDoubles) b.resize(1);

    numbersTestDoubles.resize(numbersBatchSize);
    for (auto&& b : numbersBatchDoubles) b.resize(1);

    numbersTestIntegers.resize(numbersBatchSize);
    for (auto&& b : numbersBatchDoubles) b.resize(1);
}


bool train()
{
    initNumbers();

    model m;
    

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