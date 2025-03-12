import numpy as np

#Activation functions. TODO add some more!
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Dropout function
def dropout(x, rate):
    mask = (np.random.rand(*x.shape) > rate) / (1.0 - rate)
    return x * mask, mask

#Convolutional layer
class Conv2D:
    def __init__(self, num_filters, filter_size, learning_rate=0.01):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input):
        self.input = input
        h, w = input.shape
        output_size = h - self.filter_size + 1
        self.output = np.zeros((self.num_filters, output_size, output_size))
        for f in range(self.num_filters):
            for i in range(output_size):
                for j in range(output_size):
                    region = input[i:i+self.filter_size, j:j+self.filter_size]
                    self.output[f, i, j] = np.sum(region * self.filters[f])
        return relu(self.output)

    def backward(self, d_out):
        d_filters = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(d_out.shape[1]):
                for j in range(d_out.shape[2]):
                    d_filters[f] += d_out[f, i, j] * self.input[i:i+self.filter_size, j:j+self.filter_size]
        self.filters -= self.learning_rate * d_filters

#Long short term memory
class LSTM:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, dropout_rate=0.2):
        """Initialize LSTM layer

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            learning_rate (float, optional): Step. Defaults to 0.01.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.Wf = np.random.randn(input_size, hidden_size) * 0.1
        self.Wi = np.random.randn(input_size, hidden_size) * 0.1
        self.Wo = np.random.randn(input_size, hidden_size) * 0.1
        self.Wc = np.random.randn(input_size, hidden_size) * 0.1
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))

    def forward(self, input_seq):
        self.input_seq = input_seq
        T = input_seq.shape[0]
        self.h, self.c = np.zeros((T, self.hidden_size)), np.zeros((T, self.hidden_size))
        self.f, self.i, self.o, self.c_tilde = np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h), np.zeros_like(self.h)

        for t in range(T):
            x_t = input_seq[t]
            self.f[t] = sigmoid(np.dot(x_t, self.Wf) + self.bf)
            self.i[t] = sigmoid(np.dot(x_t, self.Wi) + self.bi)
            self.o[t] = sigmoid(np.dot(x_t, self.Wo) + self.bo)
            self.c_tilde[t] = tanh(np.dot(x_t, self.Wc) + self.bc)
            self.c[t] = self.f[t] * self.c[t - 1] + self.i[t] * self.c_tilde[t]
            self.h[t] = self.o[t] * tanh(self.c[t])

        self.h, self.mask = dropout(self.h, self.dropout_rate)
        return self.h

    def backward(self, d_out):
        dWf, dWi, dWo, dWc = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wo), np.zeros_like(self.Wc)
        for t in reversed(range(d_out.shape[0])):
            dh_t = d_out[t] * self.mask[t]
            do_t = dh_t * tanh(self.c[t])
            dc_t = dh_t * self.o[t] * (1 - tanh(self.c[t]) ** 2)
            di_t = dc_t * self.c_tilde[t]
            df_t = dc_t * self.c[t - 1]
            dc_tilde_t = dc_t * self.i[t]
            dWf += np.dot(self.input_seq[t].T, df_t)
            dWi += np.dot(self.input_seq[t].T, di_t)
            dWo += np.dot(self.input_seq[t].T, do_t)
            dWc += np.dot(self.input_seq[t].T, dc_tilde_t)
        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wo -= self.learning_rate * dWo
        self.Wc -= self.learning_rate * dWc

#Fully connected layer
class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        """Initialize fully connected layer

        Args:
            input_size (_type_): _description_
            output_size (_type_): _description_
            learning_rate (float, optional): Step. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))

    #forward pass: W^T x or xW, plus the bias
    def forward(self, input):
        self.input = input
        return np.dot(input, self.W) + self.b
    
    #backpropagation: get delta for weights, subtract
    def backward(self, d_out):
        dW = np.dot(self.input.T, d_out)
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * np.sum(d_out, axis=0)

#Our CRNN model. Super lazily done, TODO OOPify this baby

class CRNN:
    def __init__(self, input_shape, num_classes):
        self.conv = Conv2D(num_filters=4, filter_size=3)
        self.lstm = LSTM(input_size=4, hidden_size=8)
        self.fc = Dense(input_size=8, output_size=num_classes)

    def forward(self, X):
        cnn_out = self.conv.forward(X)
        seq_input = cnn_out.reshape(cnn_out.shape[0], -1)
        lstm_out = self.lstm.forward(seq_input)
        output = self.fc.forward(lstm_out[-1])
        return softmax(output)

    def backward(self, d_loss):
        self.fc.backward(d_loss)
        self.lstm.backward(d_loss)
        self.conv.backward(d_loss)