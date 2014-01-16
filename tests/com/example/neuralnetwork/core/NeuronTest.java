package com.example.neuralnetwork.core;

import static org.junit.Assert.*;

import org.junit.Test;

import com.example.neuralnetwork.core.Neuron;

public class NeuronTest {

	@Test
	public void testConnect() {
		Neuron pre = new Neuron();
		Neuron post = new Neuron();
		pre.connect(post);
		assertSame(pre.getOutputSynapse(post), post.getInputSynapse(pre));
	}

}
