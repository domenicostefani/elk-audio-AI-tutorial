/*
 * This file contains the basic framework code for a JUCE plugin that uses the TensorFlow Lite library for deep inference.
 */

#include "PluginProcessor.h"

#include "PluginEditor.h"

#define MIN_SAT_GAIN 0.1f
#define MAX_SAT_GAIN 200.0f

// Load the model and init the interpreter
// Load either from a file in the filesystem or from JUCE binary data
// The second is suggested for cross-platform compatibility, as the first depends on the model being on a path that is local to the target machine
#define LOAD_MODEL_FROM_FILE 0  // If 0 load from MODEL_PATH else load from binary data
#define MODEL_PATH "/udata/model.tflite"

//==============================================================================
TFliteTemplatePluginAudioProcessor::TFliteTemplatePluginAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
    #if !JucePlugin_IsMidiEffect
        #if !JucePlugin_IsSynth
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
        #endif
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)
    #endif
                         ),
      valueTreeState(*this, nullptr, "PARAMETERS", createParameterLayout())
#endif
{
    // Resize input and output vectors so that no allocation is performet in the rt thread
    tflite_input_vec.resize(2);
    tflite_output_vec.resize(1);

    // Load the model and init the interpreter
    // Load either from a file in the filesystem or from JUCE binary data
    // The second is suggested for cross-platform compatibility, as the first depends on the model being on a path that is local to the target machine

#if (LOAD_MODEL_FROM_FILE)
    // Shortcut to avoid binary data, however it depends on local absolute path
    interpreter = InferenceEngine::createInterpreter(MODEL_PATH, true);
#else
    juce::String modelBinaryDataFilename = "saturation_model.tflite";
    // Get model index
    size_t binresource_idx = 0;
    for (int i = 0; i < BinaryData::namedResourceListSize; i++)
        if (BinaryData::originalFilenames[i] == modelBinaryDataFilename.toStdString().c_str())
            binresource_idx = i;

    const char* binNameUTF8 = BinaryData::namedResourceList[binresource_idx];

    // Get model content
    int size;
    auto model_content = BinaryData::getNamedResource(binNameUTF8, size);

    this->interpreter = InferenceEngine::createInterpreterFromBuffer(model_content, size, true);
#endif
}

TFliteTemplatePluginAudioProcessor::~TFliteTemplatePluginAudioProcessor() {
    InferenceEngine::deleteInterpreter(interpreter);
}

/** Create the parameters to add to the value tree state
 * In this case only the boolean recording state (true = rec, false = stop)
 */
AudioProcessorValueTreeState::ParameterLayout TFliteTemplatePluginAudioProcessor::createParameterLayout() {
#ifdef JUCE_ELK
    float defaultGain = 0.5f;
#else
    float defaultGain = 0.0f;
#endif
    std::vector<std::unique_ptr<RangedAudioParameter>> parameters;
    parameters.push_back(std::make_unique<AudioParameterFloat>(GAIN_ID, GAIN_NAME,
                                                               NormalisableRange<float>(0.0f,
                                                                                        1.0f,
                                                                                        0.0001,
                                                                                        0.4,
                                                                                        false),
                                                               defaultGain));

    return {parameters.begin(), parameters.end()};
}

//==============================================================================
const juce::String TFliteTemplatePluginAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool TFliteTemplatePluginAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool TFliteTemplatePluginAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool TFliteTemplatePluginAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double TFliteTemplatePluginAudioProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int TFliteTemplatePluginAudioProcessor::getNumPrograms() {
    return 1;  // NB: some hosts don't cope very well if you tell them there are 0 programs,
               // so this should be at least 1, even if you're not really implementing programs.
}

int TFliteTemplatePluginAudioProcessor::getCurrentProgram() {
    return 0;
}

void TFliteTemplatePluginAudioProcessor::setCurrentProgram(int index) {
}

const juce::String TFliteTemplatePluginAudioProcessor::getProgramName(int index) {
    return {};
}

void TFliteTemplatePluginAudioProcessor::changeProgramName(int index, const juce::String& newName) {
}

//==============================================================================
void TFliteTemplatePluginAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
}

void TFliteTemplatePluginAudioProcessor::releaseResources() {
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool TFliteTemplatePluginAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    #if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
    #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

            // This checks if the input layout matches the output layout
        #if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
        #endif

    return true;
    #endif
}
#endif

void TFliteTemplatePluginAudioProcessor::updateGain() {
    inputGain = ((AudioParameterFloat*)valueTreeState.getParameter(GAIN_ID))->get();
}

void TFliteTemplatePluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    updateGain();
    tflite_input_vec[1] = inputGain * MAX_SAT_GAIN + MIN_SAT_GAIN;

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // This is the place where you'd normally do the guts of your plugin's
    // audio processing...
    // Make sure to reset the state if your inner loop is processing
    // the samples and the outer loop is handling the channels.
    // Alternatively, you can process the samples with the channels
    // interleaved by keeping the same state.
    for (int channel = 0; channel < totalNumInputChannels; ++channel) {
        auto* channelDataIn = buffer.getWritePointer(channel);
        auto* channelData = buffer.getWritePointer(channel);

        for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
            tflite_input_vec[0] = channelDataIn[sample];
            InferenceEngine::invoke(interpreter, tflite_input_vec, tflite_output_vec);
            channelData[sample] = tflite_output_vec[0];
        }
    }
}

//==============================================================================
bool TFliteTemplatePluginAudioProcessor::hasEditor() const {
    return true;  // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* TFliteTemplatePluginAudioProcessor::createEditor() {
    return new TFliteTemplatePluginAudioProcessorEditor(*this);
}

//==============================================================================
void TFliteTemplatePluginAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void TFliteTemplatePluginAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new TFliteTemplatePluginAudioProcessor();
}
