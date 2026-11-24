#include <benchmark/benchmark.h>

#include <cstring>

#include "verify.hpp"

int main(int argc, char **argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--verify") == 0)
        return calculate_matrix::verify() == 0 ? 0 : 1;

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
