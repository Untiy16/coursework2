package com.dsaiztc.ai.ann.mlp;

import java.util.Random;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.RandomMatrices;


public class MLP
{
	/**
	 * Activation function type constants
	 */
	public static final int ACTIVATION_SIGMOID = 0;
	public static final int ACTIVATION_THRESHOLD = 1;


	//Activation function
	private int activation;

	//Learning rate parameter
	private double learningRate;

	//Input weight matrix (hidden x input)
	private DenseMatrix64F w_input;

	//Hidden weight matrices (hidden x hidden) x numberOfHiddenLayers-1
	private DenseMatrix64F[] w_hidden;

	//Output weight matrix (output x hidden)
	private DenseMatrix64F w_output;

	//Input neurons' outputs (input x 1)
	private DenseMatrix64F o_input;

	//Hidden neurons' outputs (hidden x 1) x numberOfHiddenLayers
	private DenseMatrix64F[] o_hidden;

	//Output neurons' outputs (output x 1)
	private DenseMatrix64F o_output;

	/**
	 * Constructor
	 *
	 * @param input Number of input neurons
	 * @param hidden Number of neurons in each hidden layer
	 * @param output Number of output neurons
	 * @param numberOfHiddenLayers Number of hidden layers
	 * @param learningRate Learning Rate parameter (usually 1.0)
	 * @param activation Activation function
	 */
	public MLP(int input, int hidden, int numberOfHiddenLayers, int output,  double learningRate, int activation)
	{
		w_input = new DenseMatrix64F(new double[hidden][input]);
		w_hidden = new DenseMatrix64F[numberOfHiddenLayers - 1];
		for (int i = 0; i < w_hidden.length; i++)
		{
			w_hidden[i] = new DenseMatrix64F(new double[hidden][hidden]);
		}
		w_output = new DenseMatrix64F(new double[output][hidden]);

		o_input = new DenseMatrix64F(new double[input][1]);
		o_input.zero();

		o_hidden = new DenseMatrix64F[numberOfHiddenLayers];
		for (int i = 0; i < o_hidden.length; i++)
		{
			o_hidden[i] = new DenseMatrix64F(new double[hidden][1]);
			o_hidden[i].zero();
		}

		o_output = new DenseMatrix64F(new double[output][1]);
		o_output.zero();

		this.learningRate = learningRate;
		this.activation = activation;
	}

	/**
	 * Classify method
	 *
	 * @param input Vector (column) of input
	 * @return Classification output
	 */
	public double[] classify(double[] input)
	{
		if (input.length != o_input.getNumRows())
		{
			System.out.println("classify: Vector dimensions do not match");
			return null;
		}

		DenseMatrix64F inputMatrix = new DenseMatrix64F(input.length, 1);
		for (int i = 0; i < input.length; i++)
		{
			inputMatrix.unsafe_set(i, 0, input[i]);
		}
		o_input = inputMatrix.copy();

		// First synapsis: o_hidden[0] = activation(w_input�o_input)
		CommonOps.mult(w_input, o_input, o_hidden[0]);
		o_hidden[0] = activation(o_hidden[0], activation);

		// Hidden synapsis
		for (int i = 0; i < w_hidden.length; i++)
		{
			CommonOps.mult(w_hidden[i], o_hidden[i], o_hidden[i + 1]);
			o_hidden[i + 1] = activation(o_hidden[i + 1], activation);
		}

		// Output synapsis: o_output = activation(w_output�o_hidden[numberOfHiddenLayers-1])
		CommonOps.mult(w_output, o_hidden[o_hidden.length - 1], o_output);
		o_output = activation(o_output, activation);

		double[] output = new double[o_output.numRows];
		for (int i = 0; i < output.length; i++)
		{
			output[i] = o_output.unsafe_get(i, 0);
		}

		return output;
	}

	/**
	 * Training method The labeled data should be like:
	 *  -- -- 
	 *  | input1[0]   input2[0]   ... inputN[0]   | 
	 *  | input1[1]   input2[1]   ... inputN[1]   | 
	 *  |    |            |               |       |  
	 *  | input1[M-1] input2[M-1] ... inputN[M-1] |
	 *   -- --
	 * 
	 * @param input Input of the MLP [i][j] i->Input j->Number of input
	 * @param target Target output of the MLP [i][j] i->Output j->Number of output
	 * @param N Maximum number of iterations
	 * @param max_error Maximum error
	 */
	public void train(double[][] input, double[][] target, int N, double max_error)
	{
		// The number of inputs (also outputs) must be the same and the train group must have the same length
		if (input[0].length != o_input.numRows || target[0].length != o_output.numRows || input.length != target.length)
		{
			System.out.println("train: Number of inputs/outputs does not match with the MLP created or vectors haven't the same length");
			System.out.println("train: Inputs: " + input[0].length + ":" + o_input.numRows);
			System.out.println("train: Outputs: " + target[0].length + ":" + o_output.numRows);
			System.out.println("train: Length: " + input.length + ":" + target.length);
			return;
		}

		// Random initialization of the weights
		weightInitialization();

		double total_error = 0;
		DenseMatrix64F error;
		int i = 0;

		while (i < N || total_error > max_error)
		{
			total_error = 0;
			for (int k = 0; k < input.length; k++)
			{
				// Forward Pass
				error =
						forwardPass(new DenseMatrix64F(input[0].length, 1, true, input[k]), new DenseMatrix64F(target[0].length, 1, true, target[k]));
				total_error += CommonOps.elementSumAbs(error);

				// Reverse Pass
				reversePass(error);
			}
			i++;
			if(i==N/100*10)
				System.out.println("train: 10% done" );
			if(i==N/100*20)
				System.out.println("train: 20% done" );
			if(i==N/100*30)
				System.out.println("train: 30% done" );
			if(i==N/100*40)
				System.out.println("train: 40% done" );
			if(i==N/100*50)
				System.out.println("train: 50% done" );
			if(i==N/100*60)
				System.out.println("train: 60% done" );
			if(i==N/100*70)
				System.out.println("train: 70% done" );
			if(i==N/100*80)
				System.out.println("train: 80% done" );
			if(i==N/100*90)
				System.out.println("train: 90% done" );
			if(i==N/100*100)
				System.out.println("train: 100% done" );

		}
	}

	/**
	 * Forward Pass
	 * 
	 * @param input Vector (column) of first pattern
	 * @param target Vector (column) of target
	 * @return Vector (column) of errors at output
	 */
	private DenseMatrix64F forwardPass(DenseMatrix64F input, DenseMatrix64F target)
	{
		if (input.getNumRows() != o_input.getNumRows() || target.getNumRows() != o_output.getNumRows())
		{
			System.out.println("forwardPass: Vector dimensions do not match");
			return null;
		}

		// Initializing the input of the MLP
		o_input = input.copy();

		// First synapsis: o_hidden[0] = activation(w_input�o_input)
		CommonOps.mult(w_input, o_input, o_hidden[0]);
		o_hidden[0] = activation(o_hidden[0], activation);

		// Hidden synapsis
		for (int i = 0; i < w_hidden.length; i++)
		{
			CommonOps.mult(w_hidden[i], o_hidden[i], o_hidden[i + 1]);
			o_hidden[i + 1] = activation(o_hidden[i + 1], activation);
		}

		// Output synapsis: o_output = activation(w_output�o_hidden[numberOfHiddenLayers-1])
		CommonOps.mult(w_output, o_hidden[o_hidden.length - 1], o_output);
		o_output = activation(o_output, activation);

		// Error calculation
		DenseMatrix64F error2 = new DenseMatrix64F(target.numRows, target.numCols);
		CommonOps.subtract(target, o_output, error2); // (Target-Output)
		if (activation == MLP.ACTIVATION_THRESHOLD) // Threshold function
		{
			// Error = (Target-Output)
			return error2;
		}
		else
		// Sigmoid function
		{
			// Error = Output(1-Output)(Target-Output)
			DenseMatrix64F error = o_output.copy(); // Ones matrix
			CommonOps.fill(error, 1.0);
			CommonOps.subtract(error, o_output, error); // (1-Output)
			CommonOps.elementMult(o_output, error, error); // Output(1-Output)
			CommonOps.elementMult(error, error2, error); // Output(1-Output)(Target-Output)
			return error;
		}
	}

	/**
	 * Reverse Pass
	 * 
	 * @param errors Vector of errors at the output
	 */
	private void reversePass(DenseMatrix64F errors)
	{
		updateWeights(w_output, errors, o_hidden[o_hidden.length - 1]);
		errors = calculateNextErrors(errors, w_output, o_hidden[o_hidden.length - 1]);

		for (int i = o_hidden.length - 2; i >= 0; i--)
		{
			updateWeights(w_hidden[i], errors, o_hidden[i]);
			errors = calculateNextErrors(errors, w_hidden[i], o_hidden[i]);
		}

		updateWeights(w_input, errors, o_input);
	}

	/**
	 * Weight synapsis initialization (random from -1 to 1)
	 */
	private void weightInitialization()
	{
		w_input = RandomMatrices.createRandom(w_input.numRows, w_input.numCols, -1.0, 1.0, new Random());

		for (int k = 0; k < w_hidden.length; k++)
		{
			w_hidden[k] = RandomMatrices.createRandom(w_hidden[k].numRows, w_hidden[k].numCols, -1.0, 1.0, new Random());
		}

		w_output = RandomMatrices.createRandom(w_output.numRows, w_output.numCols, -1.0, 1.0, new Random());
	}

	/**
	 * Activation function of the matrix
	 * 
	 * @param matrix Matrix that will be passed to the activation function
	 * @param activation Type of the activation function: ACTIVATION_SIGMOID, ACTIVATION_THRESHOLD
	 * @return The matrix whose elements has been activated
	 */
	private DenseMatrix64F activation(DenseMatrix64F matrix, int activation)
	{
		DenseMatrix64F temp = new DenseMatrix64F(matrix.numRows, matrix.numCols);

		switch (activation)
		{
		// Sigmoid function
			case ACTIVATION_SIGMOID:
				for (int i = 0; i < matrix.numRows; i++)
				{
					for (int j = 0; j < matrix.numCols; j++)
					{
						temp.unsafe_set(i, j, 1.0 / (1.0 + Math.pow(Math.E, -matrix.unsafe_get(i, j))));
					}
				}
				break;
			// Threshold
			case ACTIVATION_THRESHOLD:
				for (int i = 0; i < matrix.numRows; i++)
				{
					for (int j = 0; j < matrix.numCols; j++)
					{
						temp.unsafe_set(i, j, (matrix.unsafe_get(i, j) > 0.5) ? 1.0 : 0.0);
					}
				}
				break;
			default:
				temp = matrix;
				System.out.println("activation: Activation function did not work");
		}

		return temp;
	}

	/**
	 * Calculate the errors of other layers than output
	 * 
	 * @param errors The errors of the next layer (in the forward sense)
	 * @param weights Weights matrix of the next layer (in the forward sense)
	 * @param outputs Outputs of himself
	 * @return Errors in the actual layer
	 */
	private DenseMatrix64F calculateNextErrors(DenseMatrix64F errors, DenseMatrix64F weights, DenseMatrix64F outputs)
	{
		if (outputs.numRows != weights.numCols || errors.numRows != weights.numRows)
		{
			System.out.println("calculateErrors: Dimensions do not match");
			return null;
		}

		// Calculate new errors
		DenseMatrix64F error2 = new DenseMatrix64F(outputs.numRows, outputs.numCols);
		DenseMatrix64F weightsT = new DenseMatrix64F(weights.numCols, weights.numRows);
		CommonOps.transpose(weights, weightsT);
		CommonOps.mult(weightsT, errors, error2); // W'�Errors
		if (activation == MLP.ACTIVATION_THRESHOLD) // Threshold function
		{
			// NewErrors = W'�Errors
			return error2;
		}
		else
		// Sigmoid function
		{
			// NewErrors = W'�Errors*Output(1-Output)
			DenseMatrix64F error = outputs.copy(); // Ones matrix
			CommonOps.fill(error, 1.0);
			CommonOps.subtract(error, outputs, error); // (1-Output)
			CommonOps.elementMult(outputs, error, error); // Output(1-Output)
			CommonOps.elementMult(error, error2, error); // W'�Errors*Output(1-Output)
			return error;
		}
	}

	/**
	 * Updating weights (backpropagating)
	 * 
	 * @param weights Weight matrix that will be changed
	 * @param errors Next layer (of the weights matrix) errors (in the forward sense)
	 * @param outputs Previous layer (of the weights matrix) outputs (in the forward sense)
	 */
	private void updateWeights(DenseMatrix64F weights, DenseMatrix64F errors, DenseMatrix64F outputs)
	{
		double[][] errorsArray = new double[errors.numRows][outputs.numRows];
		for (int i = 0; i < errorsArray.length; i++)
		{
			for (int j = 0; j < errorsArray[0].length; j++)
			{
				errorsArray[i][j] = errors.unsafe_get(i, 0);
			}
		}
		DenseMatrix64F errorsMatrix = new DenseMatrix64F(errorsArray);

		double[][] outputsArray = new double[errors.numRows][outputs.numRows];
		for (int i = 0; i < outputsArray.length; i++)
		{
			for (int j = 0; j < outputsArray[0].length; j++)
			{
				outputsArray[i][j] = outputs.unsafe_get(j, 0);
			}
		}
		DenseMatrix64F outputsMatrix = new DenseMatrix64F(outputsArray);

		// W' = W + Errors*Outputs
		CommonOps.elementMult(errorsMatrix, outputsMatrix); // Errors*Outputs
		CommonOps.add(weights, learningRate, errorsMatrix, weights); // W + Errors*Outputs
	}

	// Main
	public static void main(String[] args)
	{
		//  logica: 00 01 10 11,
		double[][] trainArray = new double[10][2];
		trainArray[0][0] = 0;
		trainArray[0][1] = 0; //

		trainArray[1][0] = 0;
		trainArray[1][1] = 1;

		trainArray[2][0] = 1;
		trainArray[2][1] = 0;

		trainArray[3][0] = 1;
		trainArray[3][1] = 1;
		
		trainArray[4][0] = 2;
		trainArray[4][1] = 2;

		trainArray[5][0] = 3;
		trainArray[5][1] = 3;

		trainArray[6][0] = 7;
		trainArray[6][1] = 2;

		trainArray[7][0] = 1;
		trainArray[7][1] = 2;

		trainArray[8][0] = 2;
		trainArray[8][1] = 3;

		trainArray[9][0] = 4;
		trainArray[9][1] = 5;



		Random r = new Random();
		
		// sample size
		int n = 1000;

		double[][] input = new double[n][2];
		double[][] target = new double[n][1];

		for (int i = 0; i < n; i++)
		{
			input[i] = trainArray[r.nextInt(10)]; // int nextInt(int n)
			int answer = 0;
			if (input[i][0] > input[i][1]) {
				answer = 1;
			}
			target[i] = new double[] { answer };


		}

		// 2 inputs, X hidden neurons, 1 output, X hidden layer, 1.0 learning rate, activation function
		MLP NeuralNetwork = new MLP(2, 3, 3, 1, 0.7, MLP.ACTIVATION_SIGMOID);
		// Input vector, target vector, minimum operations, maximum error
		NeuralNetwork.train(input, target, 1000, 0.1);


		double[][] testArray = new double[5][2];
		testArray[0][0] = 1;
		testArray[0][1] = 2; //

		testArray[1][0] = 2;
		testArray[1][1] = 5;

		testArray[2][0] = 7;
		testArray[2][1] = 2;

		testArray[3][0] = 5;
		testArray[3][1] = 1;

		testArray[4][0] = 3;
		testArray[4][1] = 3;


		for (int i = 0; i < 5; i++)
		{
			double answer = NeuralNetwork.classify(testArray[i])[0];
			String answerStr = String.format("%(.5f", answer);
			System.out.println("Classifying: " + "[" + (int)testArray[i][0] + "," + (int)testArray[i][1] +"]"+ " Output: " + answerStr);
		}
	}
}