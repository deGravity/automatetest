import torch
from functools import reduce

class EuclideanMap(torch.nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        code_dim,
        hidden_size,
        num_layers,
        nonlin = torch.nn.functional.relu,
        use_tanh = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.code_dim = code_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_dim + code_dim

        self.decoder = ImplicitDecoder(
            input_size = self.input_size,
            output_size = self.output_dim,
            hidden_size = self.hidden_size,
            input_spacing = 0,
            nonlin = nonlin,
            use_tanh = use_tanh
        )
    def forward(self, x):
        return self.decoder(x)

class ImplicitDecoder(torch.nn.Module):
    def __init__(self,
        input_size = 2,
        output_size = 4,
        hidden_size = 1024,
        num_layers = 4,
        input_spacing = 0,
        nonlin = torch.nn.functional.relu,
        use_tanh = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_spacing = input_spacing
        self.nonlin = nonlin
        self.use_tanh = use_tanh

        for layer in range(0, self.num_layers):
            if layer == 0:
                in_size = self.input_size
            elif self.input_spacing == 0 or (layer % self.input_spacing == 0):
                in_size = self.input_size + self.hidden_size
            else:
                in_size = self.hidden_size
            if layer == self.num_layers - 1:
                out_size = self.output_size
            else:
                out_size = self.hidden_size

            setattr(
                self,
                "layer_" + str(layer),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(in_size, out_size))
            )
    
    def forward(self, x):
        y = x
        for layer in range(self.num_layers):
            if ((self.input_spacing == 0) or (layer % self.input_spacing == 0)) and (layer != 0):
                y = torch.cat([y, x], dim=1)
            y = getattr(self, "layer_" + str(layer))(y)
            if layer < (self.num_layers - 1):
                y = self.nonlin(y)
            elif self.use_tanh:
                y = torch.tanh(y)
        return y
    

# SDF Functions for Testing
class Rectangle:
    def __init__(self, *dims):
        self.dims = dims
    def sdf(self, p):
        b = 0.5*torch.tensor(self.dims, device=p.device, dtype=p.dtype)
        d = torch.abs(p) - b
        outside = torch.linalg.norm(torch.maximum(d, torch.zeros_like(d)), dim=1)
        inside = torch.clamp(torch.max(d, dim=1).values, max=0.)
        return outside + inside
    def __repr__(self):
        return f'Rectangle({self.dims})'

class Circle:
    def __init__(self, r, c):
        self.r = r
        self.c = c
    def sdf(self, p):
        center = torch.tensor(self.c, device=p.device, dtype=p.dtype)
        radius = torch.tensor(self.r, device=p.device, dtype=p.dtype)
        center_dist = torch.linalg.norm(p - center, dim=1)
        return center_dist - radius
    def __repr__(self) -> str:
        return f'Circle({self.r}, {self.c}'

class Union:
    def __init__(self, *shapes):
        self.shapes = shapes
    def sdf(self, p):
        sdfs = map(lambda x: x.sdf(p), self.shapes)
        return reduce(torch.minimum, sdfs)
    def __repr__(self):
        args = ', '.join(map(lambda x: x.__repr__(), self.shapes))
        return f'Union({args})'

class Intersection:
    def __init__(self, *shapes):
        self.shapes = shapes
    def sdf(self, p):
        sdfs = map(lambda x: x.sdf(p), self.shapes)
        return reduce(torch.maximum, sdfs)
    def __repr__(self):
        args = ', '.join(map(lambda x: x.__repr__(), self.shapes))
        return f'Intersection({args})'

class Complement:
    def __init__(self, shape):
        self.shape = shape
    def sdf(self, p):
        return -self.shape.sdf(p)
    def __repr__(self):
        return f'Complement({self.shape.__repr__()})'

class Difference:
    def __init__(self, A, *Bs):
        B = Union(*Bs)
        self.shape = Intersection(A, Complement(B))
    def sdf(self, p):
        return self.shape.sdf(p)
    def __repr__(self):
        return self.shape.__repr__()

class Translate:
    def __init__(self, shape, t):
        self.shape = shape
        self.t = t
    def sdf(self, p):
        T = torch.tensor(self.t, device=p.device, dtype=p.dtype)
        return self.shape.sdf(p - T)
    def __repr__(self):
        return f'Translate({self.shape.__repr__()}, {self.t})'

class Scale:
    def __init__(self, shape, s):
        self.shape = shape
        self.s = s
    def sdf(self, p):
        return self.shape.sdf(p / self.s)
    def __repr__(self):
        return f'Scale({self.shape.__repr__()}, {self.s})'
