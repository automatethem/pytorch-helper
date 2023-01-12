import torch

class LazilyInitializedLinear(torch.nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.layer = None
        self.out_features = out_features

    def forward(self, x):
        if not self.layer:
            in_features = x.shape[1]
            self.layer = torch.nn.Linear(in_features=in_features, out_features=self.out_features)
        x = self.layer(x)
        return x
    
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
        
class RNNLastHiddenState(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x[0] #모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, 
        hidden_state = x[1] #마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([8, 380, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 8, 32])
        x = output[:,-1]
        #print(x)
        x = hidden_state[-1]
        #print(x)
        #print(x.shape) #torch.Size([8, 32])
        return x

class LSTMLastHiddenState(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x[0] #모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, 
        hidden_state, cell_state = x[1] #마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([8, 380, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 8, 32])
        x = output[:,-1]
        #print(x)
        x = hidden_state[-1]
        #print(x)
        #print(x.shape) #torch.Size([8, 32])
        return x

class GRULastHiddenState(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = x[0] #모든 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태, 
        hidden_state = x[1] #마지막 타임 스텝(토큰: 숫자, 문자, 단어)의 숨은 상태
        #print(output.shape) #torch.Size([8, 380, 32]) 
        #print(hidden_state.shape) #torch.Size([1, 8, 32])
        x = output[:,-1]
        #print(x)
        x = hidden_state[-1]
        #print(x)
        #print(x.shape) #torch.Size([8, 32])
        return x

class DictToParameters(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(**x)

class SelectFromArray(torch.nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
