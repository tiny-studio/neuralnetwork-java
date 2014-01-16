package com.example.neuralnetwork.hierarchal.backpropagation;

import static org.junit.Assert.*;

import org.junit.Test;

import com.example.neuralnetwork.core.Neuron;
import com.example.neuralnetwork.core.PassThroughFunction;
import com.example.neuralnetwork.core.TransferFunctionStrategy;
import com.example.neuralnetwork.hierarchal.HierarchalNeuralNetwork;
import com.example.neuralnetwork.hierarchal.backpropagation.Backpropagator;

public class BackpropagatorTest {
	private static final double DELTA = 0.000000000001;

	@Test
	public void testSample() {
		TransferFunctionStrategy functionStrategy = new PassThroughFunction();
		HierarchalNeuralNetwork network = new HierarchalNeuralNetwork(
				new Neuron[] { new Neuron(functionStrategy),
						new Neuron(functionStrategy) }, new Neuron[] {
						new Neuron(), new Neuron() },
				new Neuron[] { new Neuron() });

		Neuron[] sensoryLayer = network.getLayers()[0];
		sensoryLayer[0].getOutputSynapses()[0].setWeight(0.1);
		sensoryLayer[0].getOutputSynapses()[1].setWeight(0.234);
		sensoryLayer[1].getOutputSynapses()[0].setWeight(-0.2);
		sensoryLayer[1].getOutputSynapses()[1].setWeight(0.123);

		Neuron[] associateLayer = network.getLayers()[1];
		associateLayer[0].getOutputSynapses()[0].setWeight(-0.123);
		associateLayer[1].getOutputSynapses()[0].setWeight(0.1);

		Neuron[] responseLayer = network.getLayers()[2];

		Backpropagator backpropagator = new Backpropagator(network);
		backpropagator
				.learn(new double[][][] { { { 0, 1 }, { 1 } },
						{ { 1, 1 }, { 0 } }, { { 1, 0 }, { 1 } },
						{ { 0, 0 }, { 0 } } });

		assertEquals(2.24089413758955,
				sensoryLayer[0].getOutputSynapses()[0].getWeight(), DELTA);
		assertEquals(5.610786934041788,
				sensoryLayer[0].getOutputSynapses()[1].getWeight(), DELTA);

		assertEquals(2.2208661997727495,
				sensoryLayer[1].getOutputSynapses()[0].getWeight(), DELTA);
		assertEquals(5.125745563834208,
				sensoryLayer[1].getOutputSynapses()[1].getWeight(), DELTA);

		assertEquals(-3.17798013416249, associateLayer[0].getBias(), DELTA);
		assertEquals(-1.6327273846433468, associateLayer[1].getBias(), DELTA);

		assertEquals(-5.3412305443889645,
				associateLayer[0].getOutputSynapses()[0].getWeight(), DELTA);
		assertEquals(5.400880608587149,
				associateLayer[1].getOutputSynapses()[0].getWeight(), DELTA);

		assertEquals(-2.3846121378010734, responseLayer[0].getBias(), DELTA);

	}

}
