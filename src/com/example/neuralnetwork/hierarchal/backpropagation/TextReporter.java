package com.example.neuralnetwork.hierarchal.backpropagation;

import java.io.PrintStream;

import com.example.neuralnetwork.hierarchal.HierarchalNeuralNetwork;

public class TextReporter implements LearningReporter {

	private PrintStream stream;

	public TextReporter(PrintStream stream) {
		this.stream = stream;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see LearningReporter#report(int, HierarchalNeuralNetwork, double[][][])
	 */
	@Override
	public void report(int howManyTimes, HierarchalNeuralNetwork network,
			double[][][] patterns) {
	}
}
