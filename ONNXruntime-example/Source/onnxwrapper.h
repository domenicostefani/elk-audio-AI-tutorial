/*
 * torchscript wrapper library
 *
 */
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <string>
#include <utility>
#include <vector>

namespace InferenceEngine {

class InterpreterWrap;                   // Forward definition of the InterpreterWrap class
using InterpreterPtr = InterpreterWrap*;  // Opaque pointer for classifier object

/** Dynamically allocate an instance of a classifier object (do not use in real time threads!) */
InterpreterPtr createInterpreter(const std::string& filename, bool verbose = false);

/**
 * @brief Dynamically allocate an instance of a Interpreter object from Buffer(do not use in real time threads!)
 *
 * @param buffer Caller-owned buffer containing the model
 * @param verbose  verbose mode (to disable in real time threads)
 * @return InterpreterPtr
 */
InterpreterPtr createInterpreterFromBuffer(const char* buffer, size_t bufferSize, bool verbose = false);

/** Feed a feature array (C Array) to the model, perform inference and return the prediction */
void invoke(InterpreterPtr cls, const float featureVector[], size_t numFeatures, float outputVector[], size_t numClasses);

/**
 * @brief Feed a feature array (C++ std Array) to the model, perform inference and return the prediction
 * This function is used to invoke the interpreter on a specific input vector. The input vector is passed as a std::array<float,IN_SIZE>.
 * The function does not allocate memory, it just passes the pointer to the first element of the array with the .data() method.
 *
 * @tparam IN_SIZE
 * @tparam OUT_SIZE
 * @param inp
 * @param featureArray
 * @param outputArray
 * @return int  Classification result
 */
template <std::size_t IN_SIZE, std::size_t OUT_SIZE>
void invoke(InterpreterPtr inp, std::array<float, IN_SIZE>& featureArray, std::array<float, OUT_SIZE>& outputArray) {
    invoke(inp, featureArray.data(), (size_t)IN_SIZE, outputArray.data(), (size_t)OUT_SIZE);
}

/**
 * @brief Feed a feature array (C++ std Vector) to the model, perform inference and return the prediction
 * This function is used to invoke the interpreter on a specific input vector. The input vector is passed as a std::vector<float>.
 * This is particularly useful when the input size is not known at compile time, expecially for test code.
 * A vector of random test data can be created with the help of getModelInputSize1d and passed to this function.
 *
 * @param inp           Interpreter object
 * @param inputVector Input vector
 * @param outputVector  Output vector
 * @return int          Classification result
 */
void invoke(InterpreterPtr inp, std::vector<float>& inputVector, std::vector<float>& outputVector);



/** Free the classifier memory (do not use in real time threads) */
void deleteInterpreter(InterpreterPtr cls);

}  // namespace InferenceEngine