/*
 * This file contains the basic framework code for a JUCE plugin that uses the TensorFlow Lite library for deep inference.
 */

#pragma once

#include <JuceHeader.h>

#include "PluginProcessor.h"

typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;

class TFliteTemplatePluginAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    TFliteTemplatePluginAudioProcessorEditor(TFliteTemplatePluginAudioProcessor&);
    ~TFliteTemplatePluginAudioProcessorEditor() override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    TFliteTemplatePluginAudioProcessor& audioProcessor;

    // Gain Slider
    Slider gainSlider;
    std::unique_ptr<SliderAttachment> gainSliderAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TFliteTemplatePluginAudioProcessorEditor)
};
