from graph import *

class Value:
    def __init__(self,data,_children=(),_op="", label = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data} , label = {self.label})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data , (self,other) , "+")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data , (self,other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other , (self,),f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,),'ReLU')

        def _backward():
            self.grad += (out.data > 0)* out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)
    
    def __rmul__(self,other):
        return self * other
    
    def __truediv__(self,other):
        return self * other**-1
    
    def __rtruediv__(self,other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"
    
        
a = Value(2.0 , label = 'a')
b = Value(-3.0, label = 'b')
c = Value(10.0, label = 'c')
e = a * b ; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0,label='f')
L = d * f ; L.label = 'L'

print("ans=",L)
L.backward()

dot = draw_dot(L)
dot.render('graph_output', format='png', view=True)