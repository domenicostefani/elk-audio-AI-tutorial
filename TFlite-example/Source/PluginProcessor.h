/*
 * This file contains the basic framework code for a JUCE plugin that uses the TensorFlow Lite library for deep inference.
 */

#pragma once

#include <JuceHeader.h>

#include "tflitewrapper.h"  // Put your tflite code here

//==============================================================================
/**
 */
class TFliteTemplatePluginAudioProcessor : public juce::AudioProcessor {
public:
    //==============================================================================
    TFliteTemplatePluginAudioProcessor();
    ~TFliteTemplatePluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    InferenceEngine::InterpreterPtr interpreter;

    std::vector<float> tflite_input_vec;
    std::vector<float> tflite_output_vec;

public:
    // Gain parameter
    const String GAIN_ID = "gain", GAIN_NAME = "gain";
    juce::AudioProcessorValueTreeState valueTreeState;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    void updateGain();
    float inputGain = 0.0f;

private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TFliteTemplatePluginAudioProcessor)
};
