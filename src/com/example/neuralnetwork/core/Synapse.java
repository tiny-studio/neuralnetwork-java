package com.example.neuralnetwork.core;

public class Synapse {

	private double input = 0;
	private double weight = 1.0;

	public double getOutput() {
		return input * weight;
	}

	public double getWeight() {
		return weight;
	}

	public void setInput(double input) {
		this.input = input;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}
}
