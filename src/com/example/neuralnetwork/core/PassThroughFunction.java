package com.example.neuralnetwork.core;

public class PassThroughFunction implements TransferFunctionStrategy {

	@Override
	public double execute(double x) {
		return x;
	}

	@Override
	public double differentiated(double x) {
		return x;
	}

}
