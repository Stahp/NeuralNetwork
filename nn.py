from matrix import *

def NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_nodes= input_nodes
		self.hidden_nodes= hidden_nodes
		self.output_nodes= output_nodes

		self.weights_ih= Matrix(self.hidden_nodes, self.input_nodes)
		self.weights_ho= Matrix(self.output_nodes, self.hidden_nodes)
		self.bias_h= Matrix(self.hidden_nodes, 1)
		self.bias_o= Matrix(self.output_nodes, 1)

		self.weights_ho.randomize()
		self.weights_ih.randomize()


	def feedforward(self, input):
		
		#some shit here
		return guess