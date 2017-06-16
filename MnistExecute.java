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
		// 입력, 은닉, 출력 노드의 수
		int inputNodes = 784;
		int hiddenNodes = 200;
		int outputNodes = 10;
		
		// learning rate
		// 학습률
		double learningRate = 0.1;
		
		// create instance of neural network
		// 신경망의 인스턴스 생성
		neuralNetwork.init(inputNodes, hiddenNodes, outputNodes, learningRate);
		
		try {
			
			// train the neural network
			// 신경망 학습시키기
			
			// epochs is the number of times the training data set is used for training
			// 주기(epoch)란 학습 데이터가 학습을 위해 사용되는 횟수를 의미
			int epochs = 5;
	
			int mnistLabel = 0;
			double[] inputs = new double[inputNodes];
			double[] targets = new double[outputNodes];
	
			for (int p=0; p<epochs; p++) {

				// load the mnist training data CSV file into a BufferReader
				// mnist 학습 데이터인 csv 파일을 리스트로 불러오기
				String fileName = "C:/Users/WORK1/Downloads/mnist_train.csv";
				BufferedReader brTrain = FileUtils.readBufCsv(fileName);
				// go through all records in the training data set
				// 학습 데이터 모음 내의 모든 레코드 탐색
				while(brTrain.ready()) {
				// split the record by the ',' commas
				String[] oneRow = brTrain.readLine().split(",", -1);
				
					for(int j=0; j<inputNodes+1; j++) {
						if (j==0) {
							mnistLabel = Integer.parseInt(oneRow[j]); 
						} else {
							double input = Integer.parseInt(oneRow[j]);
							// scale and shift the inputs
							// 입력 값의 범위와 값 조정
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
			// 신경망 테스트하기

			// load the mnist test data CSV file into a list
			// mnist 테스터 데이터인 csv 파일을 리스트로 불러오기
			String fileNameTest = "C:/Users/WORK1/Downloads/mnist_test.csv";
			BufferedReader brTest = FileUtils.readBufCsv(fileNameTest);

			// scorecard for how well the network performs, initially empty
			// 신경망의 성능의 지표가 되는 성적표를 아무 값도 가지지 않도록 초기화
			int brTestSize = 0;
			while(brTest.ready()) {
				brTest.readLine();
				brTestSize++;
			}
			double[] scoreCard = new double[brTestSize];

			brTest = FileUtils.readBufCsv(fileNameTest);
			
			// go through all the records in the test data set
			// 테스트 데이터 모음 내의 모든 레코드 탐색
			int i = 0;	//scoreCard 배열 순서
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
						// 입력 값의 범위와 값 조정
						input = (input / 255.0*0.99) + 0.01;
						inputs[j-1] = input;
					}
				}
				// query the network
				// 신경망에 질의
				double[] outputs = neuralNetwork.query(inputs);
				
				// the index of the highest value corresponds to the label
				// 가장 높은 값의 인덱스는 레이블의 인덱스와 일치
				double label = 0;
				double maxOut = 0.0;
				for (int k=0; k<outputs.length; k++) {
					if (maxOut < outputs[k]) {
						maxOut = outputs[k];
						label = k;
					}
				}

				// append correct or incorrect to list
				// 정답 또는 오답을 리스트에 추가
				if (label == correctLabel) {
					// network's answer matches correct answer, add 1 to scorecard
					// 정답인 경우 성적표에 1을 더함
					scoreCard[i] = 1;
				} else {
					// network's answer doesn't match correct answer, add 0 to scorecard
					// 정답이 아닌 경우 성적표에 0을 더함
					scoreCard[i] = 0;
				}
				i++;
			}
			
			// calculate the performance score, the fraction of correct answers
			// 정답의 비율인 성적을 계산해 출력
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
