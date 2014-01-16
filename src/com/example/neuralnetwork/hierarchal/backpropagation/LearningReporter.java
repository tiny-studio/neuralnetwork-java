package com.example.neuralnetwork.hierarchal.backpropagation;

import com.example.neuralnetwork.hierarchal.HierarchalNeuralNetwork;

public interface LearningReporter {

	/**
	 * 
	 * @param howManyTimes
	 * @param network
	 * @param patterns
	 */
	public abstract void report(int howManyTimes,
			HierarchalNeuralNetwork network, double[][][] patterns);

}