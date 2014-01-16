package com.example.neuralnetwork.core;

public class SigmoidFunction implements TransferFunctionStrategy {
	public SigmoidFunction() {
	}

	@Override
	public double execute(double x) {
		return (1 / (1 + Math.exp(-(x))));
	}

	@Override
	public double differentiated(double outputValue) {

		return outputValue * (1.0 - outputValue);
	}

}
