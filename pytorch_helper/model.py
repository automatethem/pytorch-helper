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

        self.layer = torch.nn.Sequential(
            torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first) #토큰의 시쿼스 입력 -> 토큰의 시퀀스 출력, input_size: 입력 토큰의 차원, hidden_size: 출력 토큰의 차원
        )
        
    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, hidden_state = self.layer(x) #output: 모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, hidden_state: 마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([32, 54, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 32, 32]) 
        #x = output[:,-1]
        x = hidden_state[0]
        #print(x.shape) #torch.Size([32, 32]) 
        return x
    
class LSTMLastOutput(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, batch_first=True):
        super().__init__()
        #print(input_size) #300
        #print(hidden_size) #32
        self.layer = torch.nn.Sequential(
            torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first) #토큰의 시쿼스 입력 -> 토큰의 시퀀스 출력, input_size: 입력 토큰의 차원, hidden_size: 출력 토큰의 차원
        )
 
    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, (hidden_state, cell_state) = self.layer(x) #output: 모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, hidden_state: 마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([32, 54, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 32, 32])
        #print(cell_state.shape) #torch.Size([1, 32, 32])
        #x = output[:,-1]
        x = hidden_state[0]
        #print(x.shape) #torch.Size([32, 32]) 
        return x

class GRULastOutput(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, batch_first=True):
        super().__init__()
        #print(input_size) #300
        #print(hidden_size) #32
        self.layer = torch.nn.Sequential(
            torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first) #토큰의 시쿼스 입력 -> 토큰의 시퀀스 출력, input_size: 입력 토큰의 차원, hidden_size: 출력 토큰의 차원
        )
            
    def forward(self, x):
        #print(x.shape) #torch.Size([32, 54, 300])
        output, (hidden_state, cell_state) = self.layer(x) #output: 모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, hidden_state: 마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([32, 54, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 32, 32])
        #print(cell_state.shape) #torch.Size([1, 32, 32])
        #x = output[:,-1]
        x = hidden_state[0]
        #print(x.shape) #torch.Size([32, 32]) 
        return x
