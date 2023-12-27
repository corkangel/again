#include "model.h"


int nothing()
{
    model m;
    return 1;
}

int layers()
{
    model m;
    layer* l = m.AddInputLayer(2);
    l = m.AddDenseLayer(1, ActivationFunction::Sigmoid, l);
    l = m.AddDenseLayer(3, ActivationFunction::Relu, l);
    
    return 1;
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
    printf("tests end\n");
    return 1;
}