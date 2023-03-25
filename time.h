#pragma once
#include <chrono>

inline int64_t epoch_usec()
{
    using namespace std::chrono;
    auto ts = high_resolution_clock::now().time_since_epoch();
    auto tsusec = duration_cast<microseconds>(ts);
    return tsusec.count();
}