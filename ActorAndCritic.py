import numpy as np


scale1 = np.sqrt(2.0 / 3) 
scale2 = np.sqrt(2.0 / 64)

np.random.seed(42)

class Actor:
    def __init__(self):
        self.W1 = np.random.randn(3, 64) * scale1
        self.W2 = np.random.randn(64, 64) * scale2
        self.W3_mean = np.random.randn(64, 1) * scale2
        self.W3_std = np.random.randn(64, 1) * scale2

        self.b1 = np.zeros((1, 64)) 
        self.b2 = np.zeros((1, 64))

    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        

        self.s = state
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        
        mean = self.a2 @ self.W3_mean
        std = np.exp(self.a2 @ self.W3_std)

        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        action = np.random.normal(mean, std)
        action = np.clip(action, -2.0, 2.0)
        log_prob = -0.5 * ((action - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
        return action, log_prob
    
    def backward(self, d_mean, d_std, lr):
        d_W3_mean = self.a2.T @ d_mean
        d_W3_std = self.a2.T @ d_std

        d_a2_mean = d_mean @ self.W3_mean.T 
        d_a2_std = d_std @ self.W3_std.T 

        d_a2 = d_a2_mean + d_a2_std

        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W2.T 

        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = self.s.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3_mean -= lr * d_W3_mean
        self.W3_std -= lr * d_W3_std





class Critic:
    def __init__(self):
        self.W1 = np.random.randn(3, 64) * scale1
        self.W2 = np.random.randn(64, 64) * scale2
        self.W3 = np.random.randn(64, 1) * scale2

        self.b1 = np.zeros((1, 64))
        self.b2 = np.zeros((1, 64)) 
        self.b3 = np.zeros((1, 1))   
    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        self.s = state
        self.z1 = state @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        return self.a2 @ self.W3 + self.b3

    def backward(self, states, returns, lr):
        values = self.forward(states)
        d_loss = 2 * (values.squeeze() - returns) / len(returns)
        d_loss = d_loss.reshape(-1, 1) #this is batch, 1

        d_W3 = self.a2.T @ d_loss
        d_b3 = np.sum(d_loss, axis=0, keepdims=True)
        d_a2 = d_loss @ self.W3.T 

        d_z2 = d_a2 * (1 - self.a2 ** 2)
        d_W2 = self.a1.T @ d_z2
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        d_a1 = d_z2 @ self.W2.T

        d_z1 = d_a1 * (1 - self.a1 ** 2)
        d_W1 = self.s.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
