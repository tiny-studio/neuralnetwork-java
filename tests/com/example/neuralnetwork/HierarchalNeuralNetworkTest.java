package com.example.neuralnetwork;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import org.junit.Test;

import com.example.neuralnetwork.core.Neuron;
import com.example.neuralnetwork.core.PassThroughFunction;
import com.example.neuralnetwork.core.SigmoidFunction;
import com.example.neuralnetwork.core.TransferFunctionStrategy;
import com.example.neuralnetwork.hierarchal.HierarchalNeuralNetwork;

public class HierarchalNeuralNetworkTest {

	private static final double DELTA = 0.000000000001;

	@Test
	public void testConnectNeuronsToNeurons() {
		Neuron[] neurons = new Neuron[5];
		for (int i = 0; i < neurons.length; i++) {
			neurons[i] = new Neuron();
		}
		HierarchalNeuralNetwork network = new HierarchalNeuralNetwork(
				new Neuron[] { neurons[0], neurons[1] }, new Neuron[] {
						neurons[2], neurons[3] }, new Neuron[] { neurons[4] });
		assertSame(neurons[0].getOutputSynapse(neurons[2]),
				neurons[2].getInputSynapse(neurons[0]));
		assertSame(neurons[0].getOutputSynapse(neurons[3]),
				neurons[3].getInputSynapse(neurons[0]));

		assertSame(network.getInputSynapses().get(0),
				neurons[0].getInputSynapse(network));
		assertEquals(network.getOutputs()[0], neurons[4].getOutput(), DELTA);
	}

	@Test
	public void testDefaultOutput() {
		Neuron[] neurons = new Neuron[5];
		for (int i = 0; i < neurons.length; i++) {
			neurons[i] = new Neuron();
		}
		HierarchalNeuralNetwork network = new HierarchalNeuralNetwork(
				new Neuron[] { neurons[0], neurons[1] }, new Neuron[] {
						neurons[2], neurons[3] }, new Neuron[] { neurons[4] });
		network.setInputs(0.f, 1.f);

		SigmoidFunction sigmoidFunction = new SigmoidFunction();
		double input_1 = sigmoidFunction.execute(0);
		double input_2 = sigmoidFunction.execute(1);

		double hidden_1 = sigmoidFunction.execute(input_1 + input_2);
		double hidden_2 = sigmoidFunction.execute(input_1 + input_2);

		double response = sigmoidFunction.execute(hidden_1 + hidden_2);

		assertEquals(response, network.getOutputs()[0], DELTA);
		assertEquals(network.getOutputs()[0],
				network.getLayers()[2][0].getOutput(), DELTA);
	}

	@Test
	public void testConnect() {
		TransferFunctionStrategy passThroughFunction = new PassThroughFunction();
		SigmoidFunction function = new SigmoidFunction();

		HierarchalNeuralNetwork pre = new HierarchalNeuralNetwork(
				new Neuron[][] { { new Neuron(passThroughFunction) },
						{ new Neuron() }, { new Neuron() } });

		HierarchalNeuralNetwork post = new HierarchalNeuralNetwork(
				new Neuron[][] { { new Neuron(passThroughFunction) },
						{ new Neuron() }, { new Neuron() } });

		pre.connect(post);
		pre.setInputs(1.0);
		pre.run();

		double firstOutput = function.execute(1.0);
		double secondOutput = function.execute(firstOutput);
		double thirdsOutput = function.execute(secondOutput);
		double fourthOutput = function.execute(thirdsOutput);

		assertEquals(firstOutput, pre.getLayers()[1][0].getOutput(), DELTA);
		assertEquals(secondOutput, pre.getOutputs()[0], DELTA);

		post.run();
		assertEquals(thirdsOutput, post.getLayers()[1][0].getOutput(), DELTA);
		assertEquals(fourthOutput, post.getOutputs()[0], DELTA);
	}

	@Test
	public void testDesignatedConstructor() {
		HierarchalNeuralNetwork network = new HierarchalNeuralNetwork(2, 4, 3,
				5);
		assertEquals(2, network.getLayers()[0].length);
		assertEquals(4, network.getLayers()[1].length);
		assertEquals(3, network.getLayers()[2].length);
		assertEquals(5, network.getLayers()[3].length);

		for (Neuron neuron : network.getInputLayer()) {
			assertEquals(true,
					neuron.getTransferFunction() instanceof PassThroughFunction);
		}

		for (int i = 1; i < network.getLayers().length; i++) {
			for (Neuron neuron : network.getLayers()[i]) {
				assertEquals(true,
						neuron.getTransferFunction() instanceof SigmoidFunction);
			}
		}

	}
}
