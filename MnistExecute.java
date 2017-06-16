package com.tradition.image.mnist;

import java.io.BufferedReader;
import java.io.IOException;

import com.util.file.FileUtils;

// Java notebook for Make Your Own Neural Network
// code for a 3-layer neural network, and code for learning the MNIST dataset
// Wonil Jung, 2017 (original python code (c) Tariq Rashid, 2016)
public class MnistExecute {
	
	private static MnistNeuralNetwork neuralNetwork = new MnistNeuralNetwork();

	public static void main(String[] args) {	

		// number of input, hidden and output nodes
		// �Է�, ����, ��� ����� ��
		int inputNodes = 784;
		int hiddenNodes = 200;
		int outputNodes = 10;
		
		// learning rate
		// �н���
		double learningRate = 0.1;
		
		// create instance of neural network
		// �Ű���� �ν��Ͻ� ����
		neuralNetwork.init(inputNodes, hiddenNodes, outputNodes, learningRate);
		
		try {
			
			// train the neural network
			// �Ű�� �н���Ű��
			
			// epochs is the number of times the training data set is used for training
			// �ֱ�(epoch)�� �н� �����Ͱ� �н��� ���� ���Ǵ� Ƚ���� �ǹ�
			int epochs = 5;
	
			int mnistLabel = 0;
			double[] inputs = new double[inputNodes];
			double[] targets = new double[outputNodes];
	
			for (int p=0; p<epochs; p++) {

				// load the mnist training data CSV file into a BufferReader
				// mnist �н� �������� csv ������ ����Ʈ�� �ҷ�����
				String fileName = "C:/Users/WORK1/Downloads/mnist_train.csv";
				BufferedReader brTrain = FileUtils.readBufCsv(fileName);
				// go through all records in the training data set
				// �н� ������ ���� ���� ��� ���ڵ� Ž��
				while(brTrain.ready()) {
				// split the record by the ',' commas
				String[] oneRow = brTrain.readLine().split(",", -1);
				
					for(int j=0; j<inputNodes+1; j++) {
						if (j==0) {
							mnistLabel = Integer.parseInt(oneRow[j]); 
						} else {
							double input = Integer.parseInt(oneRow[j]);
							// scale and shift the inputs
							// �Է� ���� ������ �� ����
							input = (input / 255.0*0.99) + 0.01;
							inputs[j-1] = input;
						}
					}
					// create the target output values (all 0.01, except the desired label which is 0.99)
					for (int i=0; i<outputNodes; i++) {
						targets[i] = 0.01;
					}
					targets[mnistLabel] = 0.99;
					
					neuralNetwork.train(inputs, targets);
				}
			}


			// test the neural network
			// �Ű�� �׽�Ʈ�ϱ�

			// load the mnist test data CSV file into a list
			// mnist �׽��� �������� csv ������ ����Ʈ�� �ҷ�����
			String fileNameTest = "C:/Users/WORK1/Downloads/mnist_test.csv";
			BufferedReader brTest = FileUtils.readBufCsv(fileNameTest);

			// scorecard for how well the network performs, initially empty
			// �Ű���� ������ ��ǥ�� �Ǵ� ����ǥ�� �ƹ� ���� ������ �ʵ��� �ʱ�ȭ
			int brTestSize = 0;
			while(brTest.ready()) {
				brTest.readLine();
				brTestSize++;
			}
			double[] scoreCard = new double[brTestSize];

			brTest = FileUtils.readBufCsv(fileNameTest);
			
			// go through all the records in the test data set
			// �׽�Ʈ ������ ���� ���� ��� ���ڵ� Ž��
			int i = 0;	//scoreCard �迭 ����
			double correctLabel = 0.0;
			while(brTest.ready()) {
				// split the record by the ',' commas
				String[] oneRow = brTest.readLine().split(",", -1);
				for(int j=0; j<inputNodes+1; j++) {
					if (j==0) {
						// correct answer is first value
						correctLabel = Integer.parseInt(oneRow[j]); 
					} else {
						double input = Integer.parseInt(oneRow[j]);
						// scale and shift the inputs
						// �Է� ���� ������ �� ����
						input = (input / 255.0*0.99) + 0.01;
						inputs[j-1] = input;
					}
				}
				// query the network
				// �Ű���� ����
				double[] outputs = neuralNetwork.query(inputs);
				
				// the index of the highest value corresponds to the label
				// ���� ���� ���� �ε����� ���̺��� �ε����� ��ġ
				double label = 0;
				double maxOut = 0.0;
				for (int k=0; k<outputs.length; k++) {
					if (maxOut < outputs[k]) {
						maxOut = outputs[k];
						label = k;
					}
				}

				// append correct or incorrect to list
				// ���� �Ǵ� ������ ����Ʈ�� �߰�
				if (label == correctLabel) {
					// network's answer matches correct answer, add 1 to scorecard
					// ������ ��� ����ǥ�� 1�� ����
					scoreCard[i] = 1;
				} else {
					// network's answer doesn't match correct answer, add 0 to scorecard
					// ������ �ƴ� ��� ����ǥ�� 0�� ����
					scoreCard[i] = 0;
				}
				i++;
			}
			
			// calculate the performance score, the fraction of correct answers
			// ������ ������ ������ ����� ���
			double performance = 0.0;
			double scoreSum = 0.0;
			for (int k=0; k<scoreCard.length; k++) {
				scoreSum = scoreSum + scoreCard[k];
			}
			performance = scoreSum / scoreCard.length;
			System.out.println("performance = " + performance);
			
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
