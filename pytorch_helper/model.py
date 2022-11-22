import torch

#Reshape((1, 28, 28))
#Reshape((28 * 28,))
#Reshape((28 * 28))
class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        if type(self.shape) == int:
            self.shape = [self.shape]

    def forward(self, x):
        return x.view((-1, *self.shape))
        
#https://en.wikipedia.org/wiki/Dot_product
class DotProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_id, movie_id):
        #print(user_id.shape) #torch.Size([1, 1, 100])
        #print(user_id.shape) #torch.Size([1, 1, 100])
        dot_product = (user_id * movie_id).sum(2)
        #print(dot_product.shape) #torch.Size([1, 1])
        return dot_product
        
class RNNLastOutput(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, batch_first=True):
        super().__init__()
        #print(input_size) #300
        #print(hidden_size) #32
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)

    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, hidden = self.rnn(x)
        #print(output.shape) #torch.Size([32, 54, 32]) #리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        #print(hidden.shape) #torch.Size([1, 32, 32]) #containing the final hidden state for each element in the batch.
        #x = output[:,-1]
        x = hidden[0]
        #print(x.shape) #torch.Size([32, 32]) #(배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        return x
    
class LSTMLastOutput(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, batch_first=True):
        super().__init__()
        #print(input_size) #300
        #print(hidden_size) #32
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
 
    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, (hidden, cell) = self.rnn(x)
        #print(output.shape) #torch.Size([32, 54, 32]) #리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        #print(hidden.shape) #torch.Size([1, 32, 32])
        #print(cell.shape) #torch.Size([1, 32, 32])
        #x = output[:,-1]
        x = hidden[0]
        #print(x.shape) #torch.Size([32, 32]) #(배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        return x

class GRULastOutput(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, batch_first=True):
        super().__init__()
        #print(input_size) #300
        #print(hidden_size) #32
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
 
    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, (hidden, cell) = self.rnn(x)
        #print(output.shape) #torch.Size([32, 54, 32]) #리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        #print(hidden.shape) #torch.Size([1, 32, 32])
        #print(cell.shape) #torch.Size([1, 32, 32])
        #x = output[:,-1]
        x = hidden[0]
        #print(x.shape) #torch.Size([32, 32]) #(배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        return x
