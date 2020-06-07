import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def dsigmoid(x):
  return sigmoid(x)* (1- sigmoid(x))

def update(num, lines_tuple, ax):
  lines= []
  for line_tuple in lines_tuple[num]:
    line, = ax.plot([],[])
    line.set_data(line_tuple[0],line_tuple[1])
    line.set_color(line_tuple[2])
    line.set_linewidth(line_tuple[3])
    lines.append(line)
  return lines
class Layer:
  def __init__(self, n_nodes):
    self.n_nodes= n_nodes
    self.values= np.zeros(shape= (n_nodes, 1))
    self.act_values= np.zeros(shape= (n_nodes, 1))
    self.weights= np.array([])
    self.bias= np.array([])

  def randomize_wb(self, prev_nodes):
    self.prev_nodes= prev_nodes
    self.weights= np.random.rand(self.n_nodes, self.prev_nodes)
    self.bias= np.random.rand(self.n_nodes, 1)

  def set_values(self, vals):
    self.act_values= np.c_[vals]

  def set_wb(self, weights, bias):
    self.weights= weights
    self.bias= bias
  
  def cal_values(self, input, activation_function= sigmoid):
    self.values= self.weights.dot(input) + self.bias
    self.act_values= np.vectorize(activation_function)(self.values)

  def __str__(self):
    string= ' Vals :\n {}\n'.format(self.act_values)
    if self.weights.size !=0:
      string  += " Weights: \n {}\nBias: \n {}\n".format(self.weights, self.bias)
    return string
class NeuralNetwork:
  def __init__(self, n_input, n_ouput, alpha= 0.5, error= 0.01, act_func= 'sigmoid'):
    self.n_input=n_input
    self.n_ouput=n_ouput
    self.alpha= alpha
    self.error= error
    if act_func== 'sigmoid':
      self.act_func= sigmoid
      self.dact_func= dsigmoid
    self.layers= [Layer(n_input), Layer(n_ouput)]

  def set_input(self, input):
    self.layers[0].set_values(input)

  def set_target(self, target):
    self.target= np.c_[target]

  def add_hidden_layer(self, n_nodes):
    self.layers.insert(-1, Layer(n_nodes))
  
  def add_hidden_layers(self, layers):
    for n_nodes in layers:
      self.add_hidden_layer(n_nodes)

  def randomize_wb(self):
    for i in range(1, len(self.layers)):
      self.layers[i].randomize_wb(self.layers[i-1].n_nodes)
  
  def set_wb(self, weights, bias):
    for i in range(1, len(self.layers)):
      self.layers[i].set_wb(weights[i-1], bias[i-1])

  def feedforward(self):
    for i in range(1, len(self.layers)):
      self.layers[i].cal_values(self.layers[i-1].act_values, activation_function= self.act_func)

  def dC_dai(self, i):
    return self.layers[-1].act_values[i] - self.target[i]

  def dai_L_daj_L_1(self, i, j, L):
    return self.dact_func(self.layers[L].values[i]) * self.layers[L].weights[i][j]

  def dai_L_dwnm_L(self, i, La, n, m, Lw):
    if La== Lw:
      if i== n:
        return self.dact_func(self.layers[La].values[i]) * self.layers[La-1].act_values[m]
      else:
        return 0
    else:
      sum= 0
      for j in range(self.layers[i-1].n_nodes):
        sum +=  self.dai_L_daj_L_1(i, j, La) * self.dai_L_dwnm_L(j, La - 1, n, m, Lw)
      return sum

  def dai_L_dbn_L(self, i, La, n, Lb):
    if La== Lb:
      if i== n:
        return self.dact_func(self.layers[La].values[i])
      else:
        return 0
    else:
      sum= 0
      for j in range(self.layers[i-1].n_nodes):
        sum +=  self.dai_L_daj_L_1(i, j, La) * self.dai_L_dbn_L(j, La - 1, n, Lb)
      return sum

  def dC_dwnm_L(self, n, m, L):
    sum= 0
    for i in range(self.layers[len(self.layers) -1].n_nodes):
      sum+= self.dC_dai(i) * self.dai_L_dwnm_L(i,len(self.layers) -1, n, m, L)
    return sum

  def dC_dbn_L(self, n, L):
    sum= 0
    for i in range(self.layers[len(self.layers) -1].n_nodes):
      sum+= self.dC_dai(i) * self.dai_L_dbn_L(i,len(self.layers) -1, n, L)
    return sum

  def backprop_test(self, n, m):
    sum= 0 
    for i in range(self.layers[-1].n_nodes):
      sum += (self.layers[-1].act_values[i] - self.target[i]) * self.dact_func(self.layers[-1].values[i]) * self.layers[-1].weights[i][n]
    sum *= self.dact_func(self.layers[-2].values[n]) * self.layers[-3].act_values[m]

    return sum, self.layers[1].weights[n][m] - self.alpha * sum
  
  def backprop(self):
    updated_w= []
    updated_b= []
    for L in range(1, len(self.layers)):
      b= []
      w= []
      for n in range(self.layers[L].n_nodes):
        db= self.dC_dbn_L(n, L)[0]
        b.append([self.layers[L].bias[n][0] - self.alpha * db])
        tmp= []
        for m in range(self.layers[L-1].n_nodes):
          dw= self.dC_dwnm_L(n, m, L)[0]
          tmp.append(self.layers[L].weights[n][m] - self.alpha * dw)
        w.append(tmp)
      updated_b.append(np.array(b))
      updated_w.append(np.array(w))
    return updated_w, updated_b

  def costFunction(self):
    sum= 0
    for i in range(self.layers[-1].n_nodes):
      sum+= (self.layers[-1].act_values[i]- self.target[i])**2
    return sum / 2
  
  def fit(self, inputs, outputs, batch_size= 32, epoch= 100):
    self.randomize_wb()
    costs= []
    updated_ws= []
    updated_bs= []
    for i in range(len(inputs)// batch_size):
      for j in range(epoch):
        wbs= []
        for k in range(batch_size*i, batch_size*(i+1)):
          self.set_input(inputs[k])
          self.set_target(outputs[k])
          self.feedforward()
          wbs.append(self.backprop())
        updated_w= np.array(wbs[0][0])
        updated_b= np.array(wbs[0][1])
        for wb in wbs[1:]:
          updated_w+= np.array(wb[0])
          updated_b+= np.array(wb[1])
        updated_w= np.divide(updated_w, len(wbs))
        updated_b= np.divide(updated_b, len(wbs))
        self.set_wb(updated_w, updated_b)
        self.feedforward()
        costs.append(self.costFunction())
        updated_ws.append(updated_w)
        updated_bs.append(updated_b)
    self.costs= costs
    self.updated_ws= updated_ws
    self.updated_bs= updated_bs

  def draw_costs(self):
    plt.plot(self.costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()

  def get_lines(self, x, y, weights):
    lines= []
    max_width= 5
    max_weight= 0
    for i in range(1, len(self.layers)):
      w= max(abs(np.max(weights[i-1])), abs(np.min(weights[i-1])))
      max_weight= w if w > max_weight else max_weight
    for i in range(1, len(self.layers)):
      for j in range(self.layers[i].n_nodes):
        for k in range(self.layers[i-1].n_nodes):
          lines.append(([x[i][j], x[i-1][k]], [y[i][j], y[i-1][k]], 'r' if weights[i-1][j][k] >0 else 'b', max_width* weights[i-1][j][k] / max_weight))
    return lines

  def draw_NN(self, animate= False):
    dis= 50
    max_rng= 0

    for layer in self.layers:
      rng= (layer.n_nodes- 1)* dis
      max_rng= rng if max_rng < rng else max_rng

    x= [[dis*i for j in range(self.layers[i].n_nodes)] for i in range(len(self.layers))]
    y= [[dis*j + (rng - (self.layers[i].n_nodes- 1)* dis) /2 for j in range(self.layers[i].n_nodes)] for i in range(len(self.layers))]

    fig = plt.figure()
    ax= plt.axes()
    if animate:
      lines_tuple= [self.get_lines(x, y, weights)  for weights in self.updated_ws]
      anim= animation.FuncAnimation(fig, update, len(lines_tuple), fargs= (lines_tuple, ax, ))
    else:
      lines= self.get_lines(x, y, self.updated_ws[-1])
      for X, Y, c, linewidth in lines:
        ax.plot(X, Y, c, linewidth= linewidth)
      for i in range(len(self.layers)):
        ax.scatter(x[i], y[i], c= 'black', s= 200)

    ax.axis('off')
    plt.show()

  def __str__(self):
    string=""
    for i in range(len(self.layers)):
      string+= "Layer {}:\n {}".format(i, self.layers[i])
    return string

nn= NeuralNetwork(2, 2)
nn.add_hidden_layers([4, 5])

inputs = [[0.05, 0.1]]
outputs= [[0.01, 0.99]]

nn.fit(inputs, outputs, 1, 100)

nn.draw_costs()

nn.draw_NN(animate= True)