/*
 * This file contains the basic framework code for a JUCE plugin that uses the ONNX runtime for deep inference.
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;

class OnnxSaturatorAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    OnnxSaturatorAudioProcessorEditor (OnnxSaturatorAudioProcessor&);
    ~OnnxSaturatorAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    OnnxSaturatorAudioProcessor& audioProcessor;

    // Gain Knob
    juce::Slider gainSlider;
    std::unique_ptr<SliderAttachment> gainSliderAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (OnnxSaturatorAudioProcessorEditor)
};
