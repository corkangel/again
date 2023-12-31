#include <cassert>

#include "model.h"

const uint32 numbersBatchSize = 20;
matrix numbersBatchDoubles;
matrix numbersBatchIntegers;
matrix numbersTestDoubles;
matrix numbersTestIntegers;

// ------------------------------ float->int classification test  ------------------------------

const uint32 numNumbers = 10;
matrix hotEncodedBatchIntegers;
matrix hotEncodedTestIntegers;

void initNumbersClassification()
{
    srand(10917917);
    numbersBatchDoubles.resize(numbersBatchSize);
    numbersBatchIntegers.resize(numbersBatchSize);
    numbersTestDoubles.resize(numbersBatchSize);
    numbersTestIntegers.resize(numbersBatchSize);
    hotEncodedBatchIntegers.resize(numbersBatchSize);
    hotEncodedTestIntegers.resize(numbersBatchSize);

    for (uint32 i=0; i < numbersBatchSize; i++)
    {
        numbersBatchDoubles[i].resize(1);
        numbersBatchDoubles[i][0] = (rand() % 1000) * 0.001 * numNumbers;

        numbersBatchIntegers[i].resize(1);
        numbersBatchIntegers[i][0] = (double)(uint32)numbersBatchDoubles[i][0];

        numbersTestDoubles[i].resize(1);
        numbersTestDoubles[i][0] = (rand() % 1000) * 0.001 * numNumbers;

        numbersTestIntegers[i].resize(1);
        numbersTestIntegers[i][0] = (double)(uint32)numbersTestDoubles[i][0];

        // hot encode the integers
        {
            auto&& c = hotEncodedBatchIntegers[i];
            c.resize(numNumbers, 0);
            c[uint32(numbersBatchIntegers[i][0])] = 1;
        }

        // hot encode the integers
        {
            auto&& c = hotEncodedTestIntegers[i];
            c.resize(numNumbers, 0);
            c[uint32(numbersTestIntegers[i][0])] = 1;
        }
    }
}

double trainNumbersClassification(model& m, uint32 numEpochs)
{
    double loss = 0;
    matrix predictions(numbersBatchSize);
    for (uint32 b=0; b < numbersBatchSize; b++)
    {
        predictions[b].resize(numNumbers);
        m.PredictSingleInput(numbersTestDoubles[b], predictions[b]);

        if (b==0)
        {
            // printf("Acts: ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", predictions[b][i]);
            // printf("\n");

            // printf("Grad: ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", m.layers.back()->gradients[i]);
            // printf("\n");

            // printf("Err:  ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", m.layers.back()->errors[i]);
            // printf("\n");            
            
            // printf("Targ: ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", hotEncodedTestIntegers[b][i]);
            // printf("\n");

        }
        //printf("%f\n", (1-predictions[b][uint32(numbersTestIntegers[b][0])]));
        loss += (1-predictions[b][uint32(numbersTestIntegers[b][0])]);
    }

    m.Train(numbersBatchDoubles, hotEncodedBatchIntegers, numEpochs, 0.5); 

    return loss / numbersBatchSize;
}

bool numbersClassification()
{
    initNumbersClassification();

    model m;

    layer* l = m.AddInputLayer(1);
    l = m.AddDenseLayer(2, ActivationFunction::Sigmoid, l);
    //l = m.AddDenseLayer(2, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(numNumbers, ActivationFunction::Softmax, l);

    for (uint32 i=0; i < 10; i++)
    {
        double loss = trainNumbersClassification(m, 100);
        printf("train numbers loss: %f %f %f\n", loss, m.loss, m.layers.back()->activationValue[6]);

        // printf("Acts: ");
        // for (int i=0; i < 10; i++)
        //     printf(" %.3f", m.layers.back()->activationValue[i]);
        // printf("\n");
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
    check("numbersClassification", numbersClassification());
    printf("tests end\n");
    return 1;
}