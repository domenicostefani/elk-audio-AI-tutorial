/*
==============================================================================*/
#include "onnxwrapper.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace InferenceEngine {

/** Function to perform the product of the elements of a vector */
template <typename T>
T vectorProduct(const std::vector<T> &v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/** Function to pretty print a vector */
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    std::stringstream stream;
    stream << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        stream << v[i];
        if (i != v.size() - 1) {
            stream << ", ";
        }
    }
    stream << "]";
    os << stream.str();
    return os;
}

// Definition of the Interpreter class
class InterpreterWrap {
public:
    /** Constructor */
    InterpreterWrap(const std::string &filename, bool verbose = false);            // Construct from file path
    InterpreterWrap(const char *buffer, size_t bufferSize, bool verbose = false);  // Construct from buffer
    void buildAndPrime(bool verbose = false);                                      // Build and prime the interpreter | Common part to the two constructors

    /** Destructor */
    ~InterpreterWrap();
    /** Internal interpreter invocation function, called by wrappers */
    void invoke_internal(const float inputVector[], size_t inputSize, float outputVector[], size_t outputSize, bool verbose = false);

    size_t inputTensorSize;
    size_t outputTensorSize;
private:
    /** Load the .onnx model and create inference session */
    Ort::Session *loadModel(const std::string &filename);
    Ort::Session *loadModelFromBuffer(const char *buffer, size_t bufferSize);

    //--------------------------------------------------------------------------
    Ort::Session *session;

    std::vector<float> inputTensorValues;
    std::vector<float> outputTensorValues;
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
};

size_t getModelInputSize1d(InterpreterPtr inp) {
    return inp->inputTensorSize;
}

size_t getModelOutputSize(InterpreterPtr inp) {
    return inp->outputTensorSize;
}

InterpreterWrap::InterpreterWrap(const std::string &filename, bool verbose) {
    // Load model
    if (verbose) {
        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;
        std::cout << "Creating environment..." << std::endl;
    }
    this->session = loadModel(filename);
    if (verbose) {
        std::cout << "Model loaded successfully." << std::endl;
        std::cout << "File: " << filename << std::endl;
    }
    buildAndPrime(verbose);
}

InterpreterWrap::InterpreterWrap(const char *buffer, size_t bufferSize, bool verbose) {
    // Load model
    if (verbose) {
        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;
        std::cout << "Creating environment..." << std::endl;
    }
    this->session = loadModelFromBuffer(buffer, bufferSize);
    if (verbose) {
        std::cout << "Model created from buffer." << std::endl;
    }
    buildAndPrime(verbose);
}

void InterpreterWrap::buildAndPrime(bool verbose) {

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    size_t numOutputNodes = session->GetOutputCount();

    const char *inputName = session->GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    const char *outputName = session->GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    if (verbose) {
        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;

        std::cout << std::endl;

        std::cout << std::left << std::setfill('.') << std::setw(30) << "Number of Input Nodes: " << std::right << std::setfill('.') << std::setw(10) << numInputNodes << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Input Name: " << std::right << std::setfill('.') << std::setw(10) << inputName << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Input Type: " << std::right << std::setfill('.') << std::setw(10) << inputType << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Input Dimensions: " << std::right << std::setfill('.') << std::setw(10) << inputDims << std::endl;

        std::cout << std::endl;

        std::cout << std::left << std::setfill('.') << std::setw(30) << "Number of Output Nodes: " << std::right << std::setfill('.') << std::setw(10) << numOutputNodes << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Output Name: " << std::right << std::setfill('.') << std::setw(10) << outputName << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Output Type: " << std::right << std::setfill('.') << std::setw(10) << outputType << std::endl;
        std::cout << std::left << std::setfill('.') << std::setw(30) << "Output Dimensions: " << std::right << std::setfill('.') << std::setw(10) << outputDims << std::endl;

        std::cout << std::endl;

        std::cout << std::setfill('-') << std::setw(40) << "" << std::endl;
    }

    inputTensorSize = vectorProduct(inputDims);
    inputTensorValues = std::vector<float>(inputTensorSize);

    outputTensorSize = vectorProduct(outputDims);
    outputTensorValues = std::vector<float>(outputTensorSize);

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    // Prime the classifier
    std::vector<float> pIv(inputTensorSize);
    std::vector<float> pOv(outputTensorSize);

    this->invoke_internal(&pIv[0], pIv.size(), &pOv[0], pOv.size());
    /*
     * The priming operation should ensure that every allocation performed
     * by the Run method is perfomed here and not in the real-time thread.
     */
}

InterpreterWrap::~InterpreterWrap() {
    delete this->session;
}

void InterpreterWrap::invoke_internal(const float inputVector[], size_t inputSize, float outputVector[], size_t outputSize, bool verbose) {
    if (inputSize != inputTensorSize)
        throw std::logic_error("Error, input vector has to have size: " + std::to_string(inputTensorSize) + " (Found " + std::to_string(inputSize) + " instead)");

    // Fill `input`.
    for (size_t i = 0; i < inputSize; ++i)
        inputTensorValues[i] = inputVector[i];

    // Run inference
    this->session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

    if (outputSize != outputTensorSize)
        throw std::logic_error("Error, output vector has to have size: " + std::to_string(outputTensorSize) + " (Found " + std::to_string(outputSize) + " instead)");

    // Copy output
    for (size_t i = 0; i < outputSize; ++i)
        outputVector[i] = outputTensorValues.at(i);
}

Ort::Session* InterpreterWrap::loadModel(const std::string &filename) {
    static Ort::Env env;  //()ORT_LOGGING_LEVEL_WARNING, "onnx-test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetOptimizedModelFilePath("optimized_model.onnx.tmp");
    return new Ort::Session(env, filename.c_str(), session_options);
}


Ort::Session* InterpreterWrap::loadModelFromBuffer(const char *buffer, size_t bufferSize) {
    static Ort::Env env;  //()ORT_LOGGING_LEVEL_WARNING, "onnx-test");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetOptimizedModelFilePath("/tmp/optimized_model.onnx.tmp");
    return new Ort::Session(env, buffer, bufferSize, session_options);
}

/***** Handle functions *****/
InterpreterPtr createInterpreter(const std::string &filename, bool verbose) {
    return new InterpreterWrap(filename, verbose);
}

InterpreterPtr createInterpreterFromBuffer(const char *buffer, size_t bufferSize, bool verbose) {
    InterpreterPtr res = new InterpreterWrap(buffer, bufferSize, verbose);
    return res;
}

void deleteInterpreter(InterpreterPtr cls) {
    if (cls)
        delete cls;
}

void invoke(InterpreterPtr cls, const float featureVector[], size_t inputSize, float outputVector[], size_t outputSize) {
    cls->invoke_internal(featureVector, inputSize, outputVector, outputSize);
}

void invoke(InterpreterPtr inp, std::vector<float> &inputVector, std::vector<float> &outputVector) {
    if (inputVector.size() != getModelInputSize1d(inp)) {
        std::cerr << "Interpreter\t|\tinvoke\t| Input vector size does not match model input size (" << inputVector.size() << " != " << getModelInputSize1d(inp) << ")" << std::endl;
        throw std::runtime_error(("Input vector size does not match model input size (" + std::to_string(inputVector.size()) + " != " + std::to_string(getModelInputSize1d(inp)) + ")").c_str());
    }
    if (outputVector.size() != getModelOutputSize(inp)) {
        std::cerr << "Interpreter\t|\tinvoke\t| Output vector size does not match model output size (" << outputVector.size() << " != " << getModelOutputSize(inp) << ")" << std::endl;
        throw std::runtime_error(("Output vector size does not match model output size (" + std::to_string(outputVector.size()) + " != " + std::to_string(getModelOutputSize(inp)) + ")").c_str());
    }
    invoke(inp, inputVector.data(), (size_t)inputVector.size(), outputVector.data(), (size_t)outputVector.size());
}

}  // namespace InferenceEngine