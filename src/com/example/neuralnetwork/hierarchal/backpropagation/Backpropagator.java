package com.example.neuralnetwork.hierarchal.backpropagation;

import java.util.HashMap;

import com.example.neuralnetwork.core.Neuron;
import com.example.neuralnetwork.core.Synapse;
import com.example.neuralnetwork.hierarchal.HierarchalNeuralNetwork;

public class Backpropagator {
	private HierarchalNeuralNetwork network;
	private double learningRate = 0.75;
	private double stabilityConstant = 0.8;
	private LearningReporter reporter;
	private HashMap<Neuron, NeuronDelta> neuronsDelta = new HashMap<>();
	private HashMap<Synapse, SynapseDelta> synapsesDelta = new HashMap<>();
	static final double ALLOWABLE_MARGIN_OF_ERROR = 0.08; // エラーの許容誤差

	public Backpropagator(HierarchalNeuralNetwork network) {
		this.network = network;
	}

	public void learn(double[][][] patterns) {
		double error = ALLOWABLE_MARGIN_OF_ERROR + 1.0; // エラーの許容誤差より大きな値

		int howManyTimes = 0;
		// 学習エラーが許容誤差内になるまで繰り返す
		while (error > ALLOWABLE_MARGIN_OF_ERROR) { // 一連の学習データを繰り返して学習する．
			for (double[][] pattern : patterns) {
				network.setInputs(pattern[0]);
				network.run();
				propagate(pattern[1]);
			}

			// 一連の学習データを繰り返して,誤差を集計する
			error = 0.0;
			for (double[][] pattern : patterns) {
				double[] inputs = pattern[0];
				double[] teachSignals = pattern[1];
				network.setInputs(inputs);
				for (int k = 0; k < network.getOutputs().length; k++) {
					error += Math.pow(
							teachSignals[k] - network.getOutputs()[k], 2.0);
				}
			}
			error *= 0.5;
			howManyTimes++;
			if (reporter != null) {
				reporter.report(howManyTimes, network, patterns);
			}
		}
	}

	public void propagate(double[] teachSignals) {
		Neuron[] responseUnits = network.getOutputLayer();
		for (int i = 0; i < responseUnits.length; i++) {
			double output = responseUnits[i].getOutput();
			double error = (teachSignals[i] - output)
					* responseUnits[i].getTransferFunction().differentiated(
							output);
			delta(responseUnits[i]).error = error;
		}

		Neuron[][] layers = network.getLayers();
		for (int i = layers.length - 2; 0 <= i; i--) {
			upldatePreviousLayer(layers[i], layers[i + 1]);
		}

		for (int i = 1; i < layers.length; i++) {// 入力層のしきい値は変更しないため1スタート
			Neuron[] layer = layers[i];
			updateBias(layer);
		}
	}

	private void updateBias(Neuron[] layer) {
		for (Neuron neuron : layer) {
			delta(neuron).biasDelta = (getLearningRate() * delta(neuron).error + getStabilityConstant()
					* delta(neuron).biasDelta);
			neuron.setBias(neuron.getBias() + delta(neuron).biasDelta);
		}
	}

	private void upldatePreviousLayer(Neuron[] presynapticNeurons,
			Neuron[] postsynapticNeurons) {

		for (Neuron presynapticNeuron : presynapticNeurons) {
			double sum = 0;

			for (Neuron postsynapticNeuron : postsynapticNeurons) {
				Synapse synapse = postsynapticNeuron
						.getInputSynapse(presynapticNeuron);

				double deltaWeight = getLearningRate()
						* delta(postsynapticNeuron).error
						* presynapticNeuron.getOutput()
						+ getStabilityConstant() * delta(synapse).weightDelta;
				delta(synapse).weightDelta = deltaWeight;

				synapse.setWeight(synapse.getWeight() + deltaWeight);

				sum += delta(postsynapticNeuron).error * synapse.getWeight();
			}
			double output = presynapticNeuron.getOutput();
			delta(presynapticNeuron).error = (presynapticNeuron
					.getTransferFunction().differentiated(output) * sum);
		}
	}

	private NeuronDelta delta(Neuron neuron) {
		if (!neuronsDelta.containsKey(neuron)) {
			neuronsDelta.put(neuron, new NeuronDelta());
		}
		return neuronsDelta.get(neuron);
	}

	private SynapseDelta delta(Synapse synapse) {
		if (!synapsesDelta.containsKey(synapse)) {
			synapsesDelta.put(synapse, new SynapseDelta());
		}
		return synapsesDelta.get(synapse);
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double eta) {
		learningRate = eta;
	}

	public double getStabilityConstant() {
		return stabilityConstant;
	}

	public void setStabilityConstant(double alpha) {
		stabilityConstant = alpha;
	}

	private static class NeuronDelta {
		private double biasDelta = 0.0;
		private double error = 0.0;
	}

	private static class SynapseDelta {
		private double weightDelta = 0.0;
	}
}
