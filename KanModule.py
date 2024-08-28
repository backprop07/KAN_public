import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


matrixA_empty = torch.tensor([[-1,3,-3,1],
                            [3,-6,3,0],
                            [-3,0,3,0],
                            [1,4,1,0]],dtype=torch.float32, requires_grad=False)/6

coef_win_empty = torch.zeros(5,5,requires_grad=False,dtype=torch.int32)

win_matrix_base = torch.tensor([0,1,2,3],requires_grad=False,dtype=torch.float)

win_matrix_empty = win_matrix_base.repeat(16).int()

x_index_empty = torch.tensor(range(64),requires_grad=False,dtype=torch.int32).contiguous()

one_empty = torch.ones(5,dtype=torch.float32, requires_grad=False)

def preactivate(x, n_int, positive=False, seq = False, causal_mask = None):
    '''
        x: 2D tensor of shape (batch_size, input_dim) or 3D tensor of shape (batch_size, seq_dim, input_dim) if seq is True
        n_int: integer, number of intervals
        positive: bool, whether to clip the input to [0,1]
        causal_mask: bool, whether to use causal mask
        reutrn: 3D tensor of shape (batch_size, input_dim, n_int+3) or 3D tensor of shape (batch_size, seq_dim, input_dim, n_int+3) if seq is True
    '''
    
    device = x.device
    global matrixA
    matrixA = matrixA_empty.to(device)
    if positive==True:
        x = x.clip(0,1)
    else:
        x = x.clip(-1,1)
    if seq == True:
        batch_size = x.shape[0]
        seq_dim = x.shape[1]
        input_dim = x.shape[2]
        output_shape = (batch_size,seq_dim,input_dim,n_int+3)
        size = batch_size*seq_dim*input_dim
    else:
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        output_shape = (batch_size,input_dim,n_int+3)
        size = batch_size*input_dim
    
    x = x.view(-1)
    #Select the adjacent control points
    if positive==True:
        gap = 1/n_int
    else:
        gap = 2/n_int
    if positive==True:
        coef_index  = torch.clip(torch.floor(x/gap).int(),max = n_int-1)
    else:
        coef_index  = torch.clip(torch.floor((x+1)/gap).int(),max = n_int-1) 
    
    global coef_win_empty
    if coef_win_empty.shape[0] < 4*size or coef_win_empty.shape[1] < n_int+3:
        coef_win_empty = torch.zeros(4*size,n_int+3,requires_grad=False, device=device,dtype=torch.float)
    coef_win = coef_win_empty[:4*size,:n_int+3].clone().to(device)
    
    global x_index_empty
    if x_index_empty.shape[0] < 4*size:
        x_index_empty = torch.tensor(range(4*size),requires_grad=False,dtype=torch.int32,device=device).contiguous()
    x_index = x_index_empty[:4*size].to(device).contiguous()
    
    global win_matrix_empty
    if win_matrix_empty.shape[0] < 4*size:
        win_matrix_empty = win_matrix_base.repeat(4*size).int().to(device)
    win_matrix = win_matrix_empty[:4*size].to(device)
    
    y_index = (coef_index.unsqueeze(1).expand(-1,4).contiguous().view(-1).int()+win_matrix).contiguous()
    
    coef_win.index_put_((x_index,y_index),torch.tensor(1,device=device,dtype=torch.float))
    
    if positive==True:
        x = (x%gap/gap).unsqueeze(-1)
    else:
        x = ((x+1)%gap/gap).unsqueeze(-1)
    
    global one_empty
    if one_empty.shape[0] < size:
        one_empty = torch.ones(size,dtype=torch.float32, requires_grad=False,device=device)
    one = one_empty[:size].to(device)
    if seq == True and causal_mask is not None:
        x = x.view(batch_size,seq_dim,input_dim,1)
        mask = causal_mask[:,:seq_dim,:input_dim].view(1,seq_dim,input_dim,1)
        x = torch.cat((torch.pow(x,3),torch.pow(x,2),x,one.view(batch_size,seq_dim,input_dim,1)),dim=-1) # (batch_size,seq_dim,input_dim,4) 
        x = x.masked_fill(mask==0,0)
        x = x.view(batch_size*seq_dim*input_dim,4).unsqueeze(1) # (size,1,4)
    else:
        one = one.view(size,1)
        x = torch.cat((torch.pow(x,3),torch.pow(x,2),x,one),dim=-1).unsqueeze(1) # (size,1,4)

    x = torch.matmul(x,matrixA)
    x = torch.matmul(x,coef_win.view(size,4,n_int+3)).view(*output_shape)
    return x

def spline_forward(x: torch.Tensor, n_int: int, coef: torch.Tensor, spline_scale: torch.Tensor = None, positive = False):
    '''
        x: 2D tensor of shape (batch_size, input_dim)
        n_int: integer, number of intervals
        coef: 3D tensor of shape (input_dim, n_int+3, output_dim) 
        spline_scale: 2D tensor of shape (input_dim, output_dim)
    '''
    batch_size = x.shape[0]
    input_dim = x.shape[1]
    output_dim = coef.shape[2]
    device = x.device
    coef = coef.to(device)
    if spline_scale is not None:
        coef = coef.permute(1,0,2)
        coef = torch.mul(coef,spline_scale.view(1,input_dim,output_dim))
        coef = coef.permute(1,0,2)
    preact  = preactivate(x, n_int, positive) # (batch_size, input_dim, n_int+3)
    preact = preact.view(batch_size,input_dim*(n_int+3))
    coef = coef.view(input_dim*(n_int+3),output_dim)
    out = torch.matmul(preact,coef)
    return out

def spline_forward_batch(x: torch.Tensor, n_int: int, coef: torch.Tensor, positive = False, causal_mask = None):
    '''
        x: 3D tensor of shape (batch_size, seq_dim, input_dim)
        n_int: integer, number of intervals
        coef: 4D tensor of shape (batch_size, input_dim, n_int+3, output_dim)
    '''
    batch_size = x.shape[0]
    seq_dim = x.shape[1]
    input_dim = x.shape[2]
    output_dim = coef.shape[3]
    device = x.device
    assert coef.shape[0] == batch_size
    assert coef.shape[1] == input_dim
    coef = coef.to(device)
    x = preactivate(x, n_int, positive, seq = True, causal_mask = causal_mask) # (batch_size, seq_dim, input_dim, n_int+3)
    x = x.view(batch_size,seq_dim,input_dim*(n_int+3))
    out = torch.matmul(x,coef.view(batch_size,input_dim*(n_int+3),output_dim))
    out = out.view(batch_size,seq_dim,output_dim)
    return out

class KanLayer_original(nn.Module):
    def __init__(self, n_int: int, input_dim: int, output_dim: int, **kwargs):
        '''
            n_int: integer, number of intervals
            input_dim: integer, input dimension
            output_dim: integer, output dimension
            **kwargs: additional keyword arguments
                coef_scale: float, the scale of the initial coefficients
                ln: bool, whether to use batch normalization before the input layer
                coef: tensor, the initial coefficients
                positive: bool, whether to clip the input to [0,1] (default: False)
        '''
        super(KanLayer_original, self).__init__()
        self.n_int = n_int
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.positive = kwargs.get('positive', False)
        self.affine = kwargs.get('affine', False)
        if self.positive:
            kwargs['norm'] = False
        if kwargs.get('norm', 'bn') == 'bn':
            self.norm_type = 'bn'
            self.norm = nn.BatchNorm1d(input_dim, affine=self.affine)
        elif kwargs.get('norm') == 'ln':
            self.norm_type = 'ln'
            self.norm = nn.LayerNorm(input_dim,elementwise_affine=self.affine)
        elif kwargs.get('norm') == False:
            self.norm_type = 'None'
        else:
            raise ValueError('Invalid norm type')
        if self.norm_type == 'ln':
            self.norm_div_term = kwargs.get('norm_div_term', math.sqrt(3))
        elif self.norm_type == 'bn':
            self.norm_div_term = kwargs.get('norm_div_term', 3)
        else:
            self.norm_div_term = 1
        
        coef_scale = kwargs.get('coef_scale', math.sqrt(6)) / math.sqrt(input_dim + output_dim)
        
        self.base_scale = nn.Parameter((torch.rand(input_dim,output_dim)-1/2)*2*coef_scale)
        self.spline_scale = nn.Parameter(torch.ones(input_dim,output_dim)*coef_scale)
        if kwargs.get('coef') is None:
            #self.coef = nn.Parameter(init_layer_coef(n_int, input_dim, output_dim, positive=self.positive))
            #self.coef = nn.Parameter(torch.randn(input_dim,n_int+3,output_dim)*coef_scale)
            self.coef = nn.Parameter((torch.rand(input_dim,n_int+3,output_dim)-1/2)*2*coef_scale)
        else:
            self.coef = kwargs.get('coef')
                
    def forward(self, x: torch.Tensor):
        '''
            x: 2D tensor of shape (batch_size, input_dim) or 3D tensor of shape (batch_size, const_dim, input_dim)
        '''
        shape = x.shape
        x = x.view(-1,shape[-1])
        if hasattr(self, 'norm'):
            x = self.norm(x)/self.norm_div_term
        spline = spline_forward(x, self.n_int, self.coef, self.spline_scale, self.positive)
        base = torch.matmul(F.silu(x),self.base_scale)
        out = spline + base
        return out.view(*shape[:-1],self.output_dim)
    
    def get_norm(self):
        norm = (self.coef[:,:-1,:]-self.coef[:,1:,:]).norm(dim=-1).var(dim = 1).sum()
        return norm
 
class KanLayer(nn.Module):
    def __init__(self, n_int: int, input_dim: int, output_dim: int, **kwargs):
        '''
            n_int: integer, number of intervals
            input_dim: integer, input dimension
            output_dim: integer, output dimension
            **kwargs: additional keyword arguments
                coef_scale: float, the scale of the initial coefficients
                ln: bool, whether to use batch normalization before the input layer
                coef: tensor, the initial coefficients
                positive: bool, whether to clip the input to [0,1] (default: False)
        '''
        super(KanLayer, self).__init__()
        self.n_int = n_int
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.positive = kwargs.get('positive', False)
        self.affine = kwargs.get('affine', False)
        if kwargs.get('norm', 'bn') == 'bn':
            self.norm_type = 'bn'
            self.norm = nn.BatchNorm1d(input_dim, affine=self.affine)
        elif kwargs.get('norm') == 'ln':
            self.norm_type = 'ln'
            self.norm = nn.LayerNorm(input_dim,elementwise_affine=self.affine)
        elif kwargs.get('norm') == False:
            self.norm_type = 'None'
        else:
            raise ValueError('Invalid norm type')
        if self.norm_type == 'ln':
            self.norm_div_term = kwargs.get('norm_div_term', math.sqrt(3))
        elif self.norm_type == 'bn':
            self.norm_div_term = kwargs.get('norm_div_term', 3)
        else:
            self.norm_div_term = 1
        
        coef_scale = kwargs.get('coef_scale', math.sqrt(6)) / math.sqrt(input_dim + output_dim)
        if kwargs.get('coef') is None:
            #self.coef = nn.Parameter(init_layer_coef(n_int, input_dim, output_dim, positive=self.positive))
            #self.coef = nn.Parameter(torch.randn(input_dim,n_int+3,output_dim)*coef_scale)
            self.coef = nn.Parameter((torch.rand(input_dim,n_int+3,output_dim)-1/2)*2*coef_scale)
        else:
            self.coef = kwargs.get('coef')
                
    def forward(self, x: torch.Tensor):
        '''
            x: 2D tensor of shape (batch_size, input_dim) or 3D tensor of shape (batch_size, const_dim, input_dim)
        '''
        shape = x.shape
        x = x.view(-1,shape[-1])
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = x/self.norm_div_term
        x = x*2-1
        x = spline_forward(x, self.n_int, self.coef, positive = self.positive)
        return x.view(*shape[:-1],self.output_dim)
    
    def get_norm(self):
        norm = (self.coef[:,:-1,:]-self.coef[:,1:,:]).norm(dim=-1).var(dim = 1).sum()
        return norm


        
