#include <cassert>

#include "model.h"

const uint32 numbersBatchSize = 20;
matrix numbersBatchDoubles;
matrix numbersBatchIntegers;
matrix numbersTestDoubles;
matrix numbersTestIntegers;

// ------------------------------ float->int classification test  ------------------------------

const uint32 numNumbers = 10;
matrix hotEncodedIntegers;

void initNumbersClassification()
{
    srand(10917917);
    numbersBatchDoubles.resize(numbersBatchSize);
    numbersBatchIntegers.resize(numbersBatchSize);
    numbersTestDoubles.resize(numbersBatchSize);
    numbersTestIntegers.resize(numbersBatchSize);
    hotEncodedIntegers.resize(numbersBatchSize);

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
        auto&& c = hotEncodedIntegers[i];
        c.resize(numNumbers, 0);
        c[uint32(numbersTestIntegers[i][0])] = 1;
    }
}

double trainNumbersClassification(model& m, uint32 numEpochs)
{
    m.Train(numbersBatchDoubles, hotEncodedIntegers, numEpochs, 0.04);

    // printf("Grad: ");
    // for (int i=0; i < 10; i++)
    //     printf(" %.3f", m.layers.back()->gradients[i]);
    // printf("\n");

    // printf("Targ: ");
    // for (int i=0; i < 10; i++)
    //     printf(" %.3f", hotEncodedIntegers[0][i]);
    // printf("\n");

    double loss = 0;
    column predictions(numNumbers);
    for (uint32 b=0; b < numbersBatchSize; b++)
    {
        m.PredictSingleInput(numbersTestDoubles[b], predictions);

        if (b==0)
        {
            printf("Acts: ");
            for (int i=0; i < 10; i++)
                printf(" %+.3f", predictions[i]);
            printf("\n");

            // printf("Grad: ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", m.layers.back()->gradients[i]);
            // printf("\n");

            // printf("Err:  ");
            // for (int i=0; i < 10; i++)
            //     printf(" %+.3f", m.layers.back()->errors[i]);
            // printf("\n");            
            
            printf("Targ: ");
            for (int i=0; i < 10; i++)
                printf(" %+.3f", hotEncodedIntegers[b][i]);
            printf("\n");

        }
        loss += pow(1-predictions[uint32(numbersTestIntegers[b][0])],2);
    }

    return loss / numbersBatchSize;
}

bool numbersClassification()
{
    initNumbersClassification();

    model m;

    layer* l = m.AddInputLayer(1);
    l = m.AddDenseLayer(3, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(numNumbers, ActivationFunction::Softmax, l);

    for (uint32 i=0; i < 10; i++)
    {
        double loss = trainNumbersClassification(m, 1);
        //printf("train numbers loss: %f %f %f\n", loss, m.loss, m.layers.back()->activationValue[6]);

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