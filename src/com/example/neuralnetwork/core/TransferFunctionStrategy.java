package com.example.neuralnetwork.core;

public interface TransferFunctionStrategy {
	TransferFunctionStrategy SIGMOID = new SigmoidFunction();

	double execute(double x);

	double differentiated(double x);
}
