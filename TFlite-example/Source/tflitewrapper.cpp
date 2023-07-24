/*
==============================================================================*/
#include "tflitewrapper.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <utility>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace InferenceEngine {

#define LOG(x) std::cerr

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

// Definition of the Interpreter class
class InterpreterWrap {
public:
    /** Constructor */
    InterpreterWrap(const std::string &filename, bool verbose = false);            // Construct from file path
    InterpreterWrap(const char *buffer, size_t bufferSize, bool verbose = false);  // Construct from buffer
    void buildAndPrime(bool verbose = false);                                      // Build and prime the interpreter | Common part to the two constructors
    /** Internal interpreter invocation function, called by wrappers */
    int invoke_internal(const float inputVector[], size_t inputSize, float outputVector[], size_t outputSize, bool verbose = false);

    int requestedInputSize() const;
    int requested2drows() const;
    int requested2dcols() const;
    int requestedOutputSize() const;

private:
    /** Step 1, TFLITE loading the .tflite model */
    std::unique_ptr<tflite::FlatBufferModel> loadModel(const std::string &filename);
    std::unique_ptr<tflite::FlatBufferModel> loadModelFromBuffer(const char *buffer, size_t bufferSize);
    /** Step 2, TFLITE building the interpreter */
    std::unique_ptr<Interpreter> buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model);

    /** ind the index of the maximum value in an array */
    int argmax(const float vec[], size_t vecSize) const;

    /** Check the input size requested by a tflite model */

    //--------------------------------------------------------------------------

    std::unique_ptr<FlatBufferModel> model;
    std::unique_ptr<Interpreter> interpreter;

    float *inputTensorPtr, *outputTensorPtr;
};

InterpreterWrap::InterpreterWrap(const std::string &filename, bool verbose) {
    // Load model
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Loading model from path: '" << filename << "'..." << std::endl;
    this->model = loadModel(filename);

    buildAndPrime(verbose);
}

InterpreterWrap::InterpreterWrap(const char *buffer, size_t bufferSize, bool verbose) {
    // Load model
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Loading model from buffer..." << std::endl;
    this->model = loadModelFromBuffer(buffer, bufferSize);

    buildAndPrime(verbose);
}

void InterpreterWrap::buildAndPrime(bool verbose) {
    // Build the interpreter
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Building interpreter..." << std::endl;
    this->interpreter = buildInterpreter(model);
    // Allocate tensor buffers.
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Allocating tensor buffers..." << std::endl;
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    // assert(interpreter != nullptr);
    // Configure the interpreter
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);

    if (interpreter == nullptr)
        throw std::runtime_error("Interpreter\t|\tconstructor\t| Failed to build interpreter. Return value is NULL.");

    if (verbose) {
        std::cout << "Interpreter\t|\tconstructor\t| Interpreter built successfully." << std::endl;
        tflite::PrintInterpreterState(interpreter.get());
    }

    // Get pointer to the input Tensor
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Getting pointer to the input tensor..." << std::endl;
    if (interpreter->inputs().size() != 1)
        throw std::runtime_error("Error, the model has more than one input tensor.  This is not supported.");
    else if (verbose) {
        std::cout << "Interpreter\t|\tconstructor\t| The model has " << interpreter->inputs().size() << " input tensors." << std::endl;
        for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
            TfLiteIntArray *input_dims = interpreter->tensor(interpreter->inputs()[i])->dims;
            // Print all sizes of the input tensor
            for (int j = 0; j < input_dims->size; ++j) {
                std::cout << "Interpreter\t|\tconstructor\t| Input tensor [" << i << "] at interpreter index " << this->interpreter->inputs()[i] << " has dimension " << j << " with size: " << input_dims->data[j] << std::endl
                          << std::flush;
            }
            // auto input_size = input_dims->data[input_dims->size - 1];
            // auto input_type = TfLiteTypeGetName(interpreter->tensor(interpreter->inputs()[i])->type);
            // std::cout << "Input tensor [" << i << "] at interpreter index " << this->interpreter->inputs()[i] << "
            // has lenth: " << input_size << " and type: " << input_type << std::endl << std::flush;
        }
    }
    this->inputTensorPtr = interpreter->typed_input_tensor<float>(
        0);  // Based on the code at
             // https://github.com/google-coral/edgetpu/blob/75e675633c2110a991426c8afa64f122b16ac372/src/cpp/examples/model_utils.cc
             // , the index used here always start from 0 and has no relation to the this->interpreter->inputs()[i]
    // assert (inputTensorPtr != nullptr);
    if (inputTensorPtr == nullptr)
        throw std::runtime_error(
            "Failed to get pointer to the input tensor.  interpreter->typed_input_tensor<float>(0) returns NULL.");

    // Get pointer to the output Tensor
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Getting pointer to the output tensor..." << std::endl;
    if (interpreter->outputs().size() != 1)
        throw std::runtime_error("Error, the model has more than one output tensor.  This is not supported.");
    else if (verbose) {
        std::cout << "Interpreter\t|\tconstructor\t| The model has " << interpreter->outputs().size() << " output tensors." << std::endl;
        for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
            TfLiteIntArray *output_dims = interpreter->tensor(interpreter->outputs()[i])->dims;
            auto output_size = output_dims->data[output_dims->size - 1];
            auto output_type = TfLiteTypeGetName(interpreter->tensor(interpreter->outputs()[i])->type);
            std::cout << "Interpreter\t|\tconstructor\t| Output tensor [" << i << "] at interpreter index " << this->interpreter->outputs()[i] << " has lenth: " << output_size << " and type: " << output_type << std::endl
                      << std::flush;
        }
    }
    this->outputTensorPtr = interpreter->typed_output_tensor<float>(
        0);  // Based on the code at
             // https://github.com/google-coral/edgetpu/blob/75e675633c2110a991426c8afa64f122b16ac372/src/cpp/examples/model_utils.cc
             // , the index used here always start from 0 and has no relation to the this->interpreter->outputs()[i]
    // assert (outputTensorPtr != nullptr);
    if (outputTensorPtr == nullptr)
        throw std::runtime_error(
            "Failed to get pointer to the output tensor.  interpreter->typed_output_tensor<float>(0) returns NULL.");

    bool prime2d = (interpreter->tensor(interpreter->inputs()[0])->dims->size == 4);
    if (verbose) {
        std::cout << "Interpreter\t|\tconstructor\t| prime2d: " << prime2d << std::endl;
        std::cout << "Interpreter\t|\tconstructor\t| interpreter->tensor(interpreter->inputs()[0])->dims->size: " << interpreter->tensor(interpreter->inputs()[0])->dims->size << std::endl;
        if (prime2d)
            std::cout << "Interpreter\t|\tconstructor\t| The model is a 2D model." << std::endl;
        else
            std::cout << "Interpreter\t|\tconstructor\t| The model is a 1D model." << std::endl;
    }
    // Prime the Interpreter
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Priming the Interpreter (Calling inference once)..." << std::endl;
    std::vector<float> pIv;
    if (prime2d) {
        if (verbose) {
            std::cout << "Interpreter\t|\tconstructor\t| Input size: [" << this->requested2drows() << " x " << this->requested2dcols() << "] | Output size: " << this->requestedOutputSize() << std::endl;
        }
        pIv.resize(this->requested2drows() * this->requested2dcols());
    } else {
        if (verbose) {
            std::cout << "Interpreter\t|\tconstructor\t| Input size: " << this->requestedInputSize() << " | Output size: " << this->requestedOutputSize() << std::endl;
        }
        pIv.resize(this->requestedInputSize());
    }
    std::vector<float> pOv;
    pOv.resize(this->requestedOutputSize());
    this->invoke_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size(), verbose);
    if (verbose)
        std::cout << "Interpreter\t|\tconstructor\t| Done.\nInterpreter\t|\tconstructor\t| Interpreter primed." << std::endl;

    /*
     * The priming operation should ensure that every allocation performed
     * by the Invoke method is perfomed here and not in the real-time thread.
     */
}

int InterpreterWrap::invoke_internal(const float inputVector[], size_t inputSize, float outputVector[], size_t outputSize, bool verbose) {
    if (verbose) {
        std::cout << "Interpreter\t|\tinvoke_internal\t| Input size: " << inputSize << " | Output size: " << outputSize << std::endl;
        std::cout << "Interpreter\t|\tinvoke_internal\t| Filling input tensor..." << std::endl
                  << std::flush;
    }
    // Fill `input`.
    for (size_t i = 0; i < inputSize; ++i) {
        this->inputTensorPtr[i] = inputVector[i];
        // if (verbose) {
        //     std::cout << "Interpreter | inputTensorPtr[" << i << "] = " << this->inputTensorPtr[i] << std::endl <<
        //     std::flush;
        // }
    }

    // Alternative with memcpy
    // memcpy(this->inputTensorPtr, inputVector, inputSize*sizeof(float));

    if (verbose)
        std::cout << "Interpreter\t|\tinvoke_internal\t| Done.\nInterpreter\t|\tinvoke_internal\t| Running inference..." << std::endl
                  << std::flush;

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    if (verbose)
        std::cout << "Interpreter\t|\tinvoke_internal\t| Done.\nInterpreter\t|\tinvoke_internal\t| Reading output tensor..." << std::endl
                  << std::flush;

    size_t requestedOutSize = requestedOutputSize();
    if (outputSize != requestedOutSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(requestedOutSize) + " (Found " + std::to_string(outputSize) + " instead)");

    if (verbose) {
        std::cout << "Interpreter\t|\tinvoke_internal\t| Done (size is OK).\nInterpreter\t|\tinvoke_internal\t| Copying to array..." << std::endl
                  << std::flush;

        for (size_t i = 0; i < outputSize; ++i)
            std::cout << "Interpreter\t|\tinvoke_internal\t| outputTensorPtr[" << i << "] :" << outputTensorPtr[i] << std::endl
                      << std::flush;
    }
    for (size_t i = 0; i < outputSize; ++i)
        outputVector[i] = outputTensorPtr[i];

    if (verbose)
        std::cout << "Interpreter\t|\tinvoke_internal\t| Done." << std::endl
                  << std::flush;

    int res = argmax(outputVector, outputSize);

    if (verbose)
        std::cout << "Interpreter\t|\tinvoke_internal\t| Done." << std::endl
                  << std::flush;
    if (verbose) {
        for (size_t i = 0; i < outputSize; ++i)
            std::cout << "Interpreter\t|\tinvoke_internal\t| outputVector[" << i << "] :" << outputVector[i] << std::endl
                      << std::flush;
    }

    return res;
}

/** STEP 1 */
std::unique_ptr<tflite::FlatBufferModel> InterpreterWrap::loadModel(const std::string &filename) {
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    TFLITE_MINIMAL_CHECK(model != nullptr);
    return model;
}

/** STEP 1 - Alternative */
std::unique_ptr<tflite::FlatBufferModel> InterpreterWrap::loadModelFromBuffer(const char *buffer, size_t bufferSize) {
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(buffer, bufferSize);
    TFLITE_MINIMAL_CHECK(model != nullptr);
    return model;
}
/** STEP 2 */
std::unique_ptr<Interpreter> InterpreterWrap::buildInterpreter(const std::unique_ptr<tflite::FlatBufferModel> &model) {
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    return interpreter;
}

int InterpreterWrap::argmax(const float vec[], size_t vecSize) const {
    float max = std::numeric_limits<float>::min();
    int argmax = -1;
    for (size_t i = 0; i < vecSize; ++i) {
        if (vec[i] > max) {
            argmax = i;
            max = vec[i];
        }
    }
    return argmax;
}

int InterpreterWrap::requestedInputSize() const {
    // get input dimension from the input tensor metadata
    // assuming one input only
    int input = this->interpreter->inputs()[0];
    TfLiteIntArray *dims = this->interpreter->tensor(input)->dims;

    int wanted_size = dims->data[1];
    return wanted_size;
}

int InterpreterWrap::requested2drows() const {
    int input = this->interpreter->inputs()[0];
    return this->interpreter->tensor(input)->dims->data[1];
}

int InterpreterWrap::requested2dcols() const {
    int input = this->interpreter->inputs()[0];
    return this->interpreter->tensor(input)->dims->data[2];
}

int InterpreterWrap::requestedOutputSize() const {
    int output_index = this->interpreter->outputs()[0];
    TfLiteIntArray *output_dims = this->interpreter->tensor(output_index)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    return output_size;
}

/***** Handle functions *****/
InterpreterPtr createInterpreter(const std::string &filename, bool verbose) {
    InterpreterPtr res = new InterpreterWrap(filename, verbose);
    return res;
}

InterpreterPtr createInterpreterFromBuffer(const char *buffer, size_t bufferSize, bool verbose) {
    InterpreterPtr res = new InterpreterWrap(buffer, bufferSize, verbose);
    return res;
}

void deleteInterpreter(InterpreterPtr inp) {
    if (inp)
        delete inp;
}

int invoke(InterpreterPtr inp, const float inputVector[], size_t inputSize, float outputVector[],
           size_t outputSize, bool verbose) {
    size_t requestedInSize = inp->requestedInputSize();
    if (inputSize != requestedInSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(requestedInSize) + " (Found " + std::to_string(inputSize) + " instead)");

    return inp->invoke_internal(inputVector, inputSize, outputVector, outputSize, verbose);
}

int invokeFlat2D(InterpreterPtr inp, const float flatFeatureMatrix[], size_t nRows, size_t nCols, float outputVector[], size_t outputSize, bool verbose) {
    size_t reqRows, reqCols;
    reqRows = inp->requested2drows();
    reqCols = inp->requested2dcols();
    if (nRows != reqRows || nCols != reqCols)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(reqRows) + "x" + std::to_string(reqCols) + " (Found " + std::to_string(nRows) + "x" + std::to_string(nCols) + " instead)");
    return inp->invoke_internal(flatFeatureMatrix, nRows * nCols, outputVector, outputSize, verbose);
}

int invoke(InterpreterPtr inp, std::vector<float> &inputVector, std::vector<float> &outputVector) {
    if (inputVector.size() != getModelInputSize1d(inp)) {
        std::cerr << "Interpreter\t|\tinvoke\t| Input vector size does not match model input size (" << inputVector.size() << " != " << getModelInputSize1d(inp) << ")" << std::endl;
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(inputVector.size()) + " != " + std::to_string(getModelInputSize1d(inp)) + ")").c_str());
    }
    if (outputVector.size() != getModelOutputSize(inp)) {
        std::cerr << "Interpreter\t|\tinvoke\t| Output vector size does not match model output size (" << outputVector.size() << " != " << getModelOutputSize(inp) << ")" << std::endl;
        throw std::runtime_error(("Output vector size does not match model output size (" + std::to_string(outputVector.size()) + " != " + std::to_string(getModelOutputSize(inp)) + ")").c_str());
    }
    return invoke(inp, inputVector.data(), (size_t)inputVector.size(), outputVector.data(), (size_t)outputVector.size());
}

int invokeFlat2D(InterpreterPtr inp, std::vector<float> &flatInputMatrix, size_t nRows, size_t nCols, std::vector<float> &outputVector, bool verbose) {
    if (flatInputMatrix.size() != nRows * nCols) {
        std::cerr << "Interpreter\t|\tinvokeFlat2D\t| Input vector size does not match the nRows and nCols values provided (" << flatInputMatrix.size() << " != " << nRows << "*" << nCols << ")" << std::endl;
        throw std::runtime_error(("Input vector size does not match the nRows and nCols values provided (" + std::to_string(flatInputMatrix.size()) + " != " + std::to_string(nRows) + "*" + std::to_string(nCols) + ")").c_str());
    }
    if ((int)nRows != inp->requested2drows())
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(nRows) + " != " + std::to_string(inp->requested2drows()) + ")").c_str());
    if ((int)nCols != inp->requested2dcols())
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(nCols) + " != " + std::to_string(inp->requested2dcols()) + ")").c_str());
    if (outputVector.size() != getModelOutputSize(inp)) {
        std::cerr << "Interpreter\t|\tinvokeFlat2D\t| Output vector size does not match model output size (" << outputVector.size() << " != " << getModelOutputSize(inp) << ")" << std::endl;
        throw std::runtime_error(("Output vector size does not match model output size (" + std::to_string(outputVector.size()) + " != " + std::to_string(getModelOutputSize(inp)) + ")").c_str());
    }

    return invokeFlat2D(inp, flatInputMatrix.data(), nRows, nCols, outputVector.data(), outputVector.size(), verbose);
}

size_t getModelInputSize1d(InterpreterPtr inp) {
    return (size_t)(inp->requestedInputSize());
}

void getModelInputSize2d(InterpreterPtr inp, size_t &rows, size_t &columns) {
    rows = (size_t)(inp->requested2drows());
    columns = (size_t)(inp->requested2dcols());
}

size_t getModelOutputSize(InterpreterPtr inp) {
    return (size_t)(inp->requestedOutputSize());
}

}  // namespace InferenceEngine