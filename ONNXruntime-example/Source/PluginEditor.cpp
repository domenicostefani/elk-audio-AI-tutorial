/*
 * This file contains the basic framework code for a JUCE plugin that uses the TensorFlow Lite library for deep inference.
 */

#include "PluginEditor.h"

#include "PluginProcessor.h"

//==============================================================================
OnnxSaturatorAudioProcessorEditor::OnnxSaturatorAudioProcessorEditor(OnnxSaturatorAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p) {
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize(300, 260);

    // Input Gain
    addAndMakeVisible(gainSlider);
    gainSlider.setSliderStyle(juce::Slider::SliderStyle::RotaryVerticalDrag);
    gainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, true, 60, 20);
    // gainSlider.setTextBoxStyle(Slider::NoTextBox, true, 0, 0);
    gainSliderAttachment.reset(new SliderAttachment(audioProcessor.valueTreeState, audioProcessor.GAIN_ID, gainSlider));
}

OnnxSaturatorAudioProcessorEditor::~OnnxSaturatorAudioProcessorEditor() {
}

//==============================================================================
void OnnxSaturatorAudioProcessorEditor::paint(juce::Graphics& g) {
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    // g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    g.fillAll(juce::Colour(0xff2E3440));
    getLookAndFeel().setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff5E81AC));
    getLookAndFeel().setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff4C566A));
    getLookAndFeel().setColour(juce::Slider::thumbColourId, juce::Colours::white);

    g.setColour(juce::Colours::white);
    g.setFont(15.0f);
    // g.drawFittedText ("TFlite Inference example\n\nSaturator", getLocalBounds(), juce::Justification::centred, 1);

    juce::Rectangle<int> area = getLocalBounds();

    juce::Rectangle<int> sliderarea = area.reduced(60);

    g.drawFittedText("Gain", sliderarea.removeFromTop(20), juce::Justification::centred, 1);
    gainSlider.setBounds(sliderarea);
}

void OnnxSaturatorAudioProcessorEditor::resized() {
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
}