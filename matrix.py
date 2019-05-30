import random

class Matrix:
	def __init__(self, rows, cols):
		self.rows= rows
		self.cols= cols
		self.data= [[0 for i in range(cols)] for j in range(rows)]

	def randomize(self):
		for i in range(self.cols):
			for j in range(self.rows):
				self.data[j][i]= random.random()*2 -1

	def add(self, m):
		for i in range(self.cols):
			for j in range(self.rows):
				self.data[j][i]+= m.data[j][i]

	def multiply(self, m):
		for