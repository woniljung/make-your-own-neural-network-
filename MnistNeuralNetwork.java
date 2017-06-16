package com.tradition.image.mnist;

public class MnistNeuralNetwork {

	private double[][] wih;
	private double[][] who;
	private int iNodes;
	private int hNodes;
	private int oNodes;
	private double lr;
	
	// initialise the neural network
	public void init(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {	

		// set number of nodes in each input, hidden, output layer
		// 입력, 은닉, 출력 계층의 노드 개수 설정
		this.iNodes = inputNodes;
		this.hNodes = hiddenNodes;
		this.oNodes = outputNodes;
		
		// link weight matrices, wih and who
		// weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		// w11 w21
		// w12 w22 etc
		// 가중치 행렬 wih와 who
		// 배열 내 가중치는 w_i_j로 표기. 노드  i에서 다음 계층의 노드 j로 연결됨을 의미
		// w11 w21
		// w12 w22 등
		this.wih = new double[hNodes][iNodes];
		this.who = new double[oNodes][hNodes];
		
		for (int i=0; i<hNodes; i++) {
			for(int j=0; j<iNodes; j++) {
				wih[i][j] = ((Math.random()-0.5D)*2.0D / Math.sqrt(iNodes));
			}
		}
		for (int i=0; i<oNodes; i++) {
			for(int j=0; j<hNodes; j++) {
				who[i][j] = ((Math.random()-0.5D)*2.0D / Math.sqrt(hNodes));
			}
		}

		// learning rate
		// 학습률
		this.lr = learningRate;
	}

	// train the neural network
	// 신경망 학습시키기
	public void train(double[] inputsList, double[] targetsList) {
		
		double[] inputs = inputsList;
		double[] targets = targetsList;
		
		double[] hiddenInputs = new double[hNodes];
		double[] hiddenOutputs = new double[hNodes];
		
		for (int i=0; i< hNodes; i++) {
			// calculate signals into hidden layer
			// 은닉 계층으로 들어오는 신호를 계산
			double hiBias = 0.0D;
			for (int j=0; j< iNodes; j++) {
				hiBias += wih[i][j]*inputs[j];
			}
			hiddenInputs[i] = hiBias;
			// calculate the signals emerging from hidden layer
			// 은닉 계층에서 나가는 신호를 계산
			hiddenOutputs[i] = this.sigmoid(hiddenInputs[i]);
		}
		double[] finalInputs = new double[oNodes];
		double[] finalOutputs = new double[oNodes];
		
		for (int i=0; i< oNodes; i++) {
			// calculate signals into final output layer
			// 최종 출력 계층으로 들어오는 신호를 계산
			double fiBias = 0.0D;
			for (int j=0; j< hNodes; j++) {
				fiBias += (who[i][j]*hiddenOutputs[j]);
			}
			finalInputs[i] = fiBias;
			// calculate the signals emerging from final ouput layer
			// 최종 출력 계층에서 나가는 신호를 계산
			finalOutputs[i] = this.sigmoid(finalInputs[i]);
		}

		// output layer error is the (target - actual)
		// 출력 계층의 오차는 (실제 값 - 계산 값)
		double[] outputErrors = new double[oNodes];
		for (int i=0; i<oNodes; i++) {
			outputErrors[i] = targets[i] - finalOutputs[i];
		}
		
		// hidden layer error is the outputErrors, split by weights, recombined at hidden nodes
		// 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
		double[] hiddenErrors = new double[hNodes];
		for (int i=0; i<hNodes; i++) {
			double errors = 0.0D;
			for(int j=0; j<oNodes; j++) {
				errors = errors + (who[j][i] * outputErrors[j]);
			}
			hiddenErrors[i] = errors;
		}
		
		// update the weights for the links between the hidden and output layers
		// 은닉 계층과 출력 계층 간의 가중치 업데이트
		double[] whoAdj = new double[oNodes];
		for (int i=0; i< oNodes; i++) {
			whoAdj[i] = outputErrors[i] * finalOutputs[i] * (1.0-finalOutputs[i]);
		}
		for (int i=0; i< oNodes; i++) {
			for (int j=0; j< hNodes; j++) {
				who[i][j] += lr * whoAdj[i] * hiddenOutputs[j];
			}
		}
		
		// update the weights for the links between the input and hidden layers
		// 입력 계층과 은닉 계층 간의 가중치 업데이트
		double[] wihAdj = new double[hNodes];
		for (int i=0; i< hNodes; i++) {
			wihAdj[i] = hiddenErrors[i] * hiddenOutputs[i] * (1.0-hiddenOutputs[i]);
		}
		for (int i=0; i< hNodes; i++) {
			for (int j=0; j< iNodes; j++) {
				wih[i][j] += lr * wihAdj[i] * inputs[j];
			}
		}
	}
	
	// query the neural network
	// 신경망에 질의하기
	public double[] query(double[] inputsList) {
		
		double[] inputs = inputsList;
		
		double[] hiddenInputs = new double[hNodes];
		double[] hiddenOutputs = new double[hNodes];
		
		for (int i=0; i< hNodes; i++) {
			// calculate signals into hidden layer
			// 은닉 계층으로 들어오는 신호를 계산
			double hiBias = 0.0D;
			for (int j=0; j< iNodes; j++) {
				hiBias += wih[i][j]*inputs[j];
			}
			hiddenInputs[i] = hiBias;
			// calculate the signals emerging from hidden layer
			// 은닉 계층에서 나가는 신호를 계산
			hiddenOutputs[i] = this.sigmoid(hiddenInputs[i]);
		}
		
		double[] finalInputs = new double[oNodes];
		double[] finalOutputs = new double[oNodes];
		for (int i=0; i< oNodes; i++) {
			// calculate signals into final output layer
			// 최종 출력 계층으로 들어오는 신호를 계산
			double fiBias = 0.0D;
			for (int j=0; j< hNodes; j++) {
				fiBias += who[i][j]*hiddenOutputs[j];
			}
			finalInputs[i] = fiBias;
			// calculate the signals emerging from final output layer
			// 최종 출력 계층에서 나가는 신호를 계산
			finalOutputs[i] = this.sigmoid(finalInputs[i]);
		}

		return finalOutputs;
	}
	
	// sigmoid function
	private double sigmoid(double x) {
		return 1.0D / (1.0D + Math.exp(-1.0D * x));
	}
	  
}
