package com.example.neuralnetwork.core;

import java.util.LinkedHashMap;

public class Neuron implements Connectable {
	private LinkedHashMap<Connectable, Synapse> inputSynapses = new LinkedHashMap<>();
	private LinkedHashMap<Connectable, Synapse> outputSynapses = new LinkedHashMap<>();
	private double bias = 0.0;
	private TransferFunctionStrategy functionStrategy = TransferFunctionStrategy.SIGMOID;;

	public Neuron() {
	}

	public Neuron(TransferFunctionStrategy functionStrategy) {
		this.functionStrategy = functionStrategy;
	}

	public void addInputSynapse(Connectable presynapticUnit, Synapse synapse) {
		inputSynapses.put(presynapticUnit, synapse);
	}

	public void addOutputSynapse(Connectable postsynapticUnit, Synapse synapse) {
		outputSynapses.put(postsynapticUnit, synapse);
	}

	public void connect(Neuron postsynapticNeuron) {
		Synapse synapse = new Synapse();
		this.outputSynapses.put(postsynapticNeuron, synapse);
		postsynapticNeuron.inputSynapses.put(this, synapse);

	}

	public double getBias() {
		return bias;
	}

	public Synapse getInputSynapse(Connectable connectable) {
		return inputSynapses.get(connectable);
	}

	public Synapse[] getInputSynapses() {
		return inputSynapses.values().toArray(new Synapse[0]);
	}

	public double getOutput() {
		double sum = 0.0;

		for (Synapse synapse : inputSynapses.values()) {
			sum += synapse.getOutput();
		}
		return functionStrategy.execute(sum + getBias());
	}

	public Synapse getOutputSynapse(Connectable connectable) {
		return outputSynapses.get(connectable);
	}

	public Synapse[] getOutputSynapses() {
		return outputSynapses.values().toArray(new Synapse[0]);
	}

	public void run() {
		for (Synapse synapse : outputSynapses.values()) {
			synapse.setInput(getOutput());
		}

	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public TransferFunctionStrategy getTransferFunction() {
		return functionStrategy;
	}
}
