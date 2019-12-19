#ifndef ACTIVATOR_H
#define ACTIVATOR_H
#include <cmath>

namespace MATHLIB{
    template<typename T>
    T sigmoid(T x){
        return T(1.0)/(T(1.0) + exp(-x));
    }

    template<typename T>
    T relu(T x){
        return x < T(0.0) ? T(0.0) : x;
    }
}

#endif