from abc import ABC, abstractmethod
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.transforms import InterpolationMode

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps_dps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.dps_scale = kwargs.get('dps_scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.dps_scale
        return x_t, norm

@register_conditioning_method(name='ps_x')
class PosteriorSamplingX(ConditioningMethod):
    def __init__(self, operator, noiser, p,**kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
        self.p = p
    # split A(x0(xt)) in A(x0(xt))=y0(yt)
    # grad on x_prev
    def grad_and_value_x_split(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_ax=y_prev+guide_coef*(self.operator.forward(x_eps, **kwargs)-y_eps)
            difference = self.operator.forward(x_prev, **kwargs)-estimation_ax
            norm = torch.linalg.norm(difference)
            norm =self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    # split A(x0(xt)) in A(x0(xt))=y0(yt) combination of A(x0(xt))=y0
    # grad on x_prev
    def grad_and_value_xandm_split(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_ax=y_prev+guide_coef*(self.operator.forward(x_eps, **kwargs)-y_eps)
            difference1 = self.operator.forward(x_prev, **kwargs)-estimation_ax
            norm1 = torch.linalg.norm(difference1)
            difference2 = self.operator.forward(x_0_hat, **kwargs)-measurement
            norm2 = torch.linalg.norm(difference2)
            norm =0.5*self.scale*norm1+0.5*self.scale*norm2
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    # directly use A(x0(xt))=y0(yt)
    # grad on x_prev
    def grad_and_value_x_nosplit(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference = y_x0-y_0_hat
            norm= torch.linalg.norm(difference)
            norm =self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]             
        return norm_grad, norm
    # combination of A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on x_prev x_0_hat respectively
    def grad_and_value_xandm_nosplit(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            coef1=kwargs['out']['coef1']
            coef2=kwargs['out']['coef2']
            coef3=kwargs['out']['coef3']
            difference0 = y_x0-y_0_hat
            norm0= torch.linalg.norm(difference0)
            difference1 = y_x0-measurement
            norm1 = torch.linalg.norm(difference1)
            norm=self.p*self.scale*norm0+(1-self.p)*self.scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        if self.noiser.__name__ == 'poisson':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference0 = y_x0-y_0_hat
            norm0= torch.linalg.norm(difference0)/measurement.abs()
            norm0 = norm0.mean()
            difference1 = y_x0-measurement
            norm1 = torch.linalg.norm(difference1)/measurement.abs()
            norm1 = norm1.mean()
            norm=self.p*self.scale*norm0+(1-self.p)*self.scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad,norm0

    def grad_and_value_y_x0_measurement(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference = y_x0-(y_0_hat+measurement)/2
            norm= torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]             
        return norm_grad, norm
    def conditioning(self, x_prev,y_prev, x_t, x_0_hat, y_0_hat,x_eps,y_eps,measurement, **kwargs):
        norm_grad, norm = self.grad_and_value_xandm_nosplit(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat,x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        x_t -= norm_grad
        return x_t, norm
    
@register_conditioning_method(name='ps_y')
class PosteriorSamplingY(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
    def grad_and_value_yandm_split(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(y_eps-self.operator.forward(x_eps, **kwargs))
            difference0 = y_prev-estimation_y
            norm0 = torch.linalg.norm(difference0)
            difference1 = measurement - y_0_hat
            norm1 = torch.linalg.norm(difference1)
            norm=self.scale*norm0+self.sub_scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]     
        return norm_grad, norm
    # no split combine A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on y_prev
    def grad_and_value_yandm_nosplit(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference0 = y_x0-y_0_hat
            norm0= torch.linalg.norm(difference0)
            difference1 = measurement - y_0_hat
            norm1 = torch.linalg.norm(difference1)
            norm=self.scale*norm0+self.sub_scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm
    #A(x0(xt))=y0 only
    # grad on y_prev
    def grad_and_value_m(self, y_prev, y_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - y_0_hat
            norm0 = torch.linalg.norm(difference)
            norm=self.sub_scale*norm0
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]
        if self.noiser.__name__ == 'poisson':   
            difference = measurement - y_0_hat
            norm0 = torch.linalg.norm(difference)/ measurement.abs()
            norm0 = norm0.mean()
            norm=self.sub_scale*norm0
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]
        return norm_grad, norm0
    # A(x0(xt))=y0(yt) only
    # grad on y_prev
    def grad_and_value_y_split(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(y_eps-self.operator.forward(x_eps, **kwargs))
            difference = y_prev-estimation_y
            norm = torch.linalg.norm(difference)
            norm=self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm
    def conditioning_ind(self, x_prev,y_prev,y_t, x_0_hat,y_0_hat, x_eps,y_eps,measurement, **kwargs):
        norm_grad, norm = self.grad_and_value_m(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        y_t -= norm_grad
        return y_t, norm
    
    def conditioning_d(self, x_prev,y_prev,y_t, y_0_hat, x_eps,y_eps,guide_coef,measurement, **kwargs):
        coef1=kwargs['out']['coef1']
        coef2=kwargs['out']['coef2']
        norm_grad, norm = self.grad_and_value_y2(x_prev=x_prev,y_prev=y_prev, y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=guide_coef,measurement=measurement, **kwargs)
        y_t -= norm_grad * self.scale
        y_0_hat=(y_t-coef2*y_prev)/coef1
        norm_grad, norm = self.grad_and_value_y1(y_prev=y_t, y_0_hat=y_0_hat, measurement=measurement, **kwargs)
        y_t -= norm_grad * self.sub_scale
        return y_t, norm
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass

@register_conditioning_method(name='ps_x_ip')
class PosteriorSamplingX(ConditioningMethod):
    def __init__(self, operator, noiser,p ,**kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
        self.p=p
    # split A(x0(xt)) in A(x0(xt))=y0(yt)
    # grad on x_prev
    def grad_and_value_x_split(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_ax=y_prev+guide_coef*(self.operator.forward(x_eps, **kwargs)-y_eps)
            difference = self.operator.forward(x_prev, **kwargs)-estimation_ax
            norm = torch.linalg.norm(difference)
            norm =self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    # split A(x0(xt)) in A(x0(xt))=y0(yt) combination of A(x0(xt))=y0
    # grad on x_prev
    def grad_and_value_xandm_split(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_ax=y_prev+guide_coef*(self.operator.forward(x_eps, **kwargs)-y_eps)
            difference1 = self.operator.forward(x_prev, **kwargs)-estimation_ax
            norm1 = torch.linalg.norm(difference1)
            difference2 = self.operator.forward(x_0_hat, **kwargs)-measurement
            norm2 = torch.linalg.norm(difference2)
            norm =0.5*self.scale*norm1+0.5*self.scale*norm2
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    # directly use A(x0(xt))=y0(yt)
    # grad on x_prev
    def grad_and_value_x_nosplit(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference = y_x0-y_0_hat
            # coef=(guide_coef+1).sqrt()
            # difference=coef*difference
            norm= torch.linalg.norm(difference)
            norm =self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]             
        return norm_grad, norm
    # combination of A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on x_prev x_0_hat respectively
    def grad_and_value_xandm_nosplit(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            coef1=kwargs['out']['coef1']
            coef2=kwargs['out']['coef2']
            coef3=kwargs['out']['coef3']
            difference0 = y_x0-y_0_hat
            norm0= torch.linalg.norm(difference0)
            difference1 = y_x0-measurement
            norm1 = torch.linalg.norm(difference1)
            norm=self.p*self.scale*norm0+(1-self.p)*self.scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad,norm1
    def grad_and_value_y_x0_measurement(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference = y_x0-(y_0_hat+measurement)/2
            norm= torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]             
        return norm_grad, norm
    def conditioning(self, x_prev,y_prev, x_t, x_0_hat, y_0_hat,x_eps,y_eps,measurement, **kwargs):
        norm_grad, norm = self.grad_and_value_xandm_nosplit(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat,x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        x_t -= norm_grad
        return x_t, norm
    


@register_conditioning_method(name='ps_y_ip')
class PosteriorSamplingY(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
    # split A(x0(xt)) combine A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on y_prev
    def grad_and_value_yandm_split(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(y_eps-self.operator.forward(x_eps, **kwargs))
            difference0 = y_prev-estimation_y
            norm0 = torch.linalg.norm(difference0)
            difference1 = measurement - y_0_hat
            norm1 = torch.linalg.norm(difference1)
            # norm=self.scale*norm0+self.sub_scale*norm1
            norm=self.scale*norm0+self.sub_scale*norm1
            # norm=self.sub_scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm
    # no split combine A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on y_prev
    def grad_and_value_yandm_nosplit(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference0 = y_x0-y_0_hat
            norm0= torch.linalg.norm(difference0)
            difference1 = measurement - y_0_hat
            norm1 = torch.linalg.norm(difference1)
            norm=self.scale*norm0+self.sub_scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm
    #A(x0(xt))=y0 only
    # grad on y_prev
    def grad_and_value_m(self, y_prev, y_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - y_0_hat
            norm0 = torch.linalg.norm(difference)
            norm=self.sub_scale*norm0
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]
        return norm_grad, norm0
    # A(x0(xt))=y0(yt) only
    # grad on y_prev
    def grad_and_value_y_split(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(y_eps-self.operator.forward(x_eps, **kwargs))
            difference = y_prev-estimation_y
            norm = torch.linalg.norm(difference)
            norm=self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm
    def conditioning_ind(self, x_prev,y_prev,y_t, x_0_hat,y_0_hat, x_eps,y_eps,measurement, **kwargs):
        norm_grad, norm = self.grad_and_value_m(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        y_t -= norm_grad
        return y_t, norm
    
    def conditioning_d(self, x_prev,y_prev,y_t, y_0_hat, x_eps,y_eps,guide_coef,measurement, **kwargs):
        coef1=kwargs['out']['coef1']
        coef2=kwargs['out']['coef2']
        norm_grad, norm = self.grad_and_value_y2(x_prev=x_prev,y_prev=y_prev, y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=guide_coef,measurement=measurement, **kwargs)
        y_t -= norm_grad * self.scale
        y_0_hat=(y_t-coef2*y_prev)/coef1
        norm_grad, norm = self.grad_and_value_y1(y_prev=y_t, y_0_hat=y_0_hat, measurement=measurement, **kwargs)
        y_t -= norm_grad * self.sub_scale
        return y_t, norm
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass

#super-resolution
@register_conditioning_method(name='ps_x_sr')
class PosteriorSamplingX(ConditioningMethod):
    def __init__(self, operator, noiser,p, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
        self.out_size = kwargs.get('out_size', 1.0)
        self.p=p

    def grad_and_value_x(self, x_prev,y_prev, x_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_ax=self.transform(y_prev)+guide_coef*(self.operator.forward(x_eps, **kwargs)-self.transform(y_eps))
            difference = self.operator.forward(x_prev, **kwargs)-estimation_ax
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
             
        return norm_grad, norm
    # directly use A(x0(xt))=y0(yt)
    # grad on x_prev
    def grad_and_value_x0(self, x_prev,y_prev, x_0_hat, y_0_hat,x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference = y_x0-self.transform(y_0_hat)
            # coef=(guide_coef+1).sqrt()
            # difference=coef*difference
            norm= torch.linalg.norm(difference)
            norm =self.scale*norm
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]             
        return norm_grad, norm
    # combination of A(x0(xt))=y0(yt) and A(x0(xt))=y0
    # grad on x_prev x_0_hat respectively
    def grad_and_value_x0toxt(self, x_prev,y_prev, x_0_hat,y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            coef1=kwargs['out']['coef1']
            coef2=kwargs['out']['coef2']
            coef3=kwargs['out']['coef3']
            difference0 = y_x0-self.transform64(y_0_hat)
            # difference0 = self.transform256(y_x0)-y_0_hat
            # difference1 = y_x0-self.operator.forward(y_0_hat, **kwargs)
            norm0= torch.linalg.norm(difference0)
            difference1 = y_x0-measurement
            norm1 = torch.linalg.norm(difference1)
            norm=self.p*self.scale*norm0+(1-self.p)*self.scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        if self.noiser.__name__ == 'poisson':
            y_x0=self.operator.forward(x_0_hat, **kwargs)
            difference0 = y_x0-self.transform64(y_0_hat)
            norm0= torch.linalg.norm(difference0)/measurement.abs()
            norm0 = norm0.mean()
            difference1 = y_x0-measurement
            norm1 = torch.linalg.norm(difference1)/measurement.abs()
            norm1 = norm1.mean()
            norm=self.p*self.scale*norm0+(1-self.p)*self.scale*norm1
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad,norm1

    def conditioning(self, x_prev,y_prev, x_t, x_0_hat, y_0_hat,x_eps,y_eps,measurement, **kwargs):
        #LANCZOS
        # self.transform64 = T.Resize((self.out_size,self.out_size),interpolation=InterpolationMode.LANCZOS)
        #BICUBIC
        self.transform64 = T.Resize((self.out_size,self.out_size),interpolation=InterpolationMode.BICUBIC)
        #bilinear
        # self.transform64 = T.Resize((self.out_size,self.out_size))
        self.transform256 = T.Resize((256,256))
        # self.transform = T.Resize((self.out_size,self.out_size))
        norm_grad, norm = self.grad_and_value_x0toxt(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat,x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        x_t -= norm_grad
        return x_t, norm

    
@register_conditioning_method(name='ps_y_sr')
class PosteriorSamplingY(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.sub_scale = kwargs.get('sub_scale', 1.0)
        self.out_size = kwargs.get('out_size', 1.0)
    #A(x0(xt))=y0 only
    # grad on y_prev
    def grad_and_value_y1(self, y_prev, y_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.transform64(y_0_hat)
            norm0 = torch.linalg.norm(difference)
            norm=self.sub_scale*norm0
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]
        if self.noiser.__name__ == 'poisson':   
            difference = measurement - self.transform64(y_0_hat)
            norm0 = torch.linalg.norm(difference)/ measurement.abs()
            norm0 = norm0.mean()
            norm=self.sub_scale*norm0
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]
        return norm_grad, norm0
        
    def grad_and_value_y0(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(self.transform(y_eps)-self.operator.forward(x_eps, **kwargs))
            difference1 = self.transform(y_prev)-estimation_y
            norm0 = torch.linalg.norm(difference1)
            difference2 = measurement - self.transform(y_0_hat)
            norm1 = torch.linalg.norm(difference2)
            norm=self.scale*norm0+self.sub_scale*(norm1)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm1
    
    def grad_and_value_y2(self, x_prev,y_prev, y_0_hat, x_eps,y_eps,guide_coef,measurement,**kwargs):
        if self.noiser.__name__ == 'gaussian':
            estimation_y=self.operator.forward(x_prev, **kwargs)+guide_coef*(self.transform(y_eps)-self.operator.forward(x_eps, **kwargs))
            difference = self.transform(y_prev)-estimation_y
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=y_prev)[0]             
        return norm_grad, norm

    def conditioning_ind(self, x_prev,y_prev,y_t, x_0_hat,y_0_hat, x_eps,y_eps,measurement, **kwargs):
        #LANCZOS
        # self.transform64 = T.Resize((self.out_size,self.out_size),interpolation=InterpolationMode.LANCZOS)
        #BICUBIC
        self.transform64 = T.Resize((self.out_size,self.out_size),interpolation=InterpolationMode.BICUBIC)
        #bilinear
        # self.transform64 = T.Resize((self.out_size,self.out_size))
        self.transform256 = T.Resize((256,256))
        norm_grad, norm = self.grad_and_value_y1(x_prev=x_prev,y_prev=y_prev, x_0_hat=x_0_hat,y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=kwargs['out']['guide_coef'],measurement=measurement, **kwargs)
        y_t -= norm_grad
        return y_t, norm

    def conditioning_d(self, x_prev,y_prev,y_t, y_0_hat, x_eps,y_eps,measurement, **kwargs):
        self.transform = T.Resize((self.out_size,self.out_size))
        guide_coef=self.transform(kwargs['out']['guide_coef'])
        norm_grad, norm = self.grad_and_value_y2(x_prev=x_prev,y_prev=y_t, y_0_hat=y_0_hat, x_eps=x_eps,y_eps=y_eps,guide_coef=guide_coef,measurement=measurement, **kwargs)
        y_t -= norm_grad * self.scale
        norm_grad, norm = self.grad_and_value_y1(y_prev=y_prev, y_0_hat=y_0_hat, measurement=measurement, **kwargs)
        y_t -= norm_grad * self.sub_scale
        return y_t, norm
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
