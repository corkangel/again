#include <vector>

typedef std::vector<double> column;
typedef std::vector<column> matrix;

using uint8 = unsigned char;
using int8 = char;
using uint16 = unsigned short;
using int16 = short;
using uint32 = unsigned int;
using int32 = int;
using uint64 = unsigned long;
using int64 = long;

int argmax(const column& values);
column softmax(const column& inputs);
