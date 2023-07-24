/*
 * TFlite Interpreter library
 * Author: Domenico Stefani (domenico.stefani96@gmail.com)
 *
 * This header exposes the functions used to run inference with TFLite.
 * To see how to use it, check test_library/main.cpp
 *
 * The structure is a bit convoluted since it used an opaque pointer (to avoid having to include many headers later)
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

class InterpreterWrap;                    // Forward definition of the Interpreter class
using InterpreterPtr = InterpreterWrap*;  // Opaque pointer for Interpreter object

/**
 * @brief Get the Model Input Size for 1dimentional input models
 *
 * @param inp
 * @return size_t
 */
size_t getModelInputSize1d(InterpreterPtr inp);

/**
 * @brief Get the Model Input Size2d object
 *
 * @param inp
 * @param rows
 * @param columns
 */
void getModelInputSize2d(InterpreterPtr inp, size_t& rows, size_t& columns);

/**
 * @brief Get the Model Output size
 *
 * @param inp
 * @return size_t
 */
size_t getModelOutputSize(InterpreterPtr inp);

/**
 * @brief Dynamically allocate an instance of a Interpreter object (do not use in real time threads!)
 *
 * @param filename path to the tflite model file
 * @param verbose  verbose mode (to disable in real time threads)
 * @return InterpreterPtr
 */
InterpreterPtr createInterpreter(const std::string& filename, bool verbose = false);

/**
 * @brief Dynamically allocate an instance of a Interpreter object from Buffer(do not use in real time threads!)
 *
 * @param buffer Caller-owned buffer containing the model
 * @param verbose  verbose mode (to disable in real time threads)
 * @return InterpreterPtr
 */
InterpreterPtr createInterpreterFromBuffer(const char* buffer, size_t bufferSize, bool verbose = false);

/**
 * @brief Free the Interpreter memory (do not use in real time threads)
 *
 * @param inp pointer to the Interpreter object
 */
void deleteInterpreter(InterpreterPtr inp);

/**
 * @brief  Feed a feature array (C Array) to the model, perform inference and return the prediction
 * The input vector is passed as a pointer to a float (c array).
 *
 * @param inp
 * @param inputVector
 * @param inputSize
 * @param outputVector
 * @param outputSize
 * @param verbose
 * @return int
 */
int invoke(InterpreterPtr inp, const float inputVector[], size_t inputSize, float outputVector[], size_t outputSize, bool verbose = false);

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
int invoke(InterpreterPtr inp, std::array<float, IN_SIZE>& featureArray, std::array<float, OUT_SIZE>& outputArray) {
    return invoke(inp, featureArray.data(), (size_t)IN_SIZE, outputArray.data(), (size_t)OUT_SIZE);
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
int invoke(InterpreterPtr inp, std::vector<float>& inputVector, std::vector<float>& outputVector);

/**
 * @brief Invoke the interpreter for a 2D matrix stored in a flat array
 *
 * @param inp
 * @param flatFeatureMatrix
 * @param nRows
 * @param nCols
 * @param outputVector
 * @param outputSize
 * @param verbose
 * @return int
 */
int invokeFlat2D(InterpreterPtr inp, const float flatFeatureMatrix[], size_t nRows, size_t nCols, float outputVector[], size_t outputSize, bool verbose = false);

/**
 * @brief Invoke the interpreter for a 2D matrix stored in a flat std::array
 *
 * @tparam N_ROWS         Number of rows in the input matrix
 * @tparam N_COLS         Number of columns in the input matrix
 * @tparam OUT_SIZE       Number of classes in the output vector
 * @param inp             Interpreter object
 * @param flatInputMatrix Input matrix
 * @param outputArray     Output vector
 * @return int            Classification result
 */
template <std::size_t N_ROWS, std::size_t N_COLS, std::size_t OUT_SIZE>
int invokeFlat2D(InterpreterPtr inp, std::array<float, N_ROWS * N_COLS>& flatInputMatrix, std::array<float, OUT_SIZE>& outputArray, bool verbose = false) {
    return invokeFlat2D(inp, flatInputMatrix.data(), N_ROWS, N_COLS, outputArray.data(), (size_t)OUT_SIZE, verbose);
}

/**
 * @brief Invoke the interpreter for a 2D matrix stored in a flat vector
 *
 * @param inp             Interpreter object
 * @param flatInputMatrix Input matrix
 * @param nRows           Number of rows in the input matrix
 * @param nCols           Number of columns in the input matrix
 * @param outputVector    Output vector
 * @return int            Classification result
 */
int invokeFlat2D(InterpreterPtr inp, std::vector<float>& flatInputMatrix, size_t nRows, size_t nCols, std::vector<float>& outputVector, bool verbose = false);

}  // namespace InferenceEngine