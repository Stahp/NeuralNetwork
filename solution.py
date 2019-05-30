from matrix import *
from nn import *

def main():
	neuralnetwork= NeuralNetwork(2, 2, 1)
	input= [1, 0]
	output= neuralnetwork.feedforward(input)

if __name__ == '__main__':
	main()