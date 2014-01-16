package com.example.neuralnetwork.hierarchal;

import java.util.ArrayList;
import java.util.List;

import com.example.neuralnetwork.core.Connectable;
import com.example.neuralnetwork.core.Neuron;
import com.example.neuralnetwork.core.PassThroughFunction;
import com.example.neuralnetwork.core.Synapse;

public class HierarchalNeuralNetwork implements Connectable {

	private Neuron[][] layers;
	private List<Synapse> inputSynapses = new ArrayList<>();

	public HierarchalNeuralNetwork(int... numberOfNeurons) {
		Neuron[][] layers = new Neuron[numberOfNeurons.length][];

		Neuron[] inputLayer = new Neuron[numberOfNeurons[0]];
		PassThroughFunction passThroughFunction = new PassThroughFunction();
		for (int j = 0; j < inputLayer.length; j++) {
			inputLayer[j] = new Neuron(passThroughFunction);
		}
		layers[0] = inputLayer;

		for (int i = 1; i < numberOfNeurons.length; i++) {
			Neuron[] previousLayer = new Neuron[numberOfNeurons[i]];
			for (int j = 0; j < previousLayer.length; j++) {
				previousLayer[j] = new Neuron();
			}
			layers[i] = previousLayer;
		}
		initialize(layers);
	}

	public HierarchalNeuralNetwork(Neuron[]... layers) {
		initialize(layers);
	}

	private void conectNeuronsToPostsynapticNeurons(Neuron[] neurons,
			Neuron[] postsynapticNeurons) {
		for (Neuron neuron : neurons) {
			for (Neuron postsynapticNeuron : postsynapticNeurons) {
				neuron.connect(postsynapticNeuron);
			}
		}
	}

	public void connect(HierarchalNeuralNetwork post) {
		Neuron[] outputLayer = getOutputLayer();
		for (int i = 0; i < post.inputSynapses.size(); i++) {
			outputLayer[i].addOutputSynapse(post, post.inputSynapses.get(i));
		}
	}

	public Neuron[] getInputLayer() {
		return this.layers[0];
	}

	public List<Synapse> getInputSynapses() {
		return inputSynapses;
	}

	public Neuron[][] getLayers() {
		return layers;
	}

	public Neuron[] getOutputLayer() {
		return layers[layers.length - 1];
	}

	public double[] getOutputs() {
		run();
		Neuron[] outputLayer = getOutputLayer();
		double[] outputs = new double[outputLayer.length];
		for (int i = 0; i < outputLayer.length; i++) {
			outputs[i] = outputLayer[i].getOutput();
		}
		return outputs;
	}

	private void initialize(Neuron[]... layers) {
		this.layers = layers;
		for (int i = 0; i < layers.length - 1; i++) {
			conectNeuronsToPostsynapticNeurons(layers[i], layers[i + 1]);
		}
		for (Neuron neuron : getInputLayer()) {
			Synapse synapse = new Synapse();
			inputSynapses.add(synapse);
			neuron.addInputSynapse(this, synapse);
		}
	}

	public void run() {
		for (Neuron[] layer : layers) {
			for (Neuron neuron : layer) {
				neuron.run();
			}
		}
	}

	public void setInputs(double... inputs) {
		for (int i = 0; i < inputSynapses.size(); i++) {
			inputSynapses.get(i).setInput(inputs[i]);
		}
	}
}
