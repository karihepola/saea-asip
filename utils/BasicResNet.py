import torch
import torch.nn as nn
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)

def clipped_relu(x):
        return torch.clamp(x, min=0.0, max=1.0)

class BasicResNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 features: int = 350, 
                 depth: int = 2,
                 num_outputs: int = 1, 
                 normalization: str = 'layernorm', 
                 activation: str ='relu'):
     
        super(BasicResNet, self).__init__()

        '''
        ResFNN architecture
    
        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        '''
     
        # First Layer
        self.first = nn.Linear(input_dim, features)
  
        # Residual layer
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
  
        # Decoder Layer
        self.last = nn.Linear(features, num_outputs)

        # Normalization Layer
        self.normalization = normalization
        if self.normalization == 'layernorm':
            self.layernorm = nn.LayerNorm(features)
        elif self.normalization == 'batchnorm':
            self.batchnorm = nn.BatchNorm1d(features)
        else:
            self.normalization = None

        # Activation Layer
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("That activation is unknown")
        
   				
    def forward(self, x):
        # First Layer
        out = self.first(x)

        # Residual Layer
        for residual in self.residuals:        
            if self.normalization == 'batchnorm':
                out = out + self.batchnorm(self.activation(residual(out)))
            elif self.normalization == 'layernorm':
                out = out + self.layernorm(self.activation(residual(out)))
            else:
                out = out + self.activation(residual(out))
        
        # Output Layer with Sigmoid activation
        out = torch.sigmoid(self.last(out))
        #out = self.last(out)
        #out = clipped_relu(self.last(out))


        return out