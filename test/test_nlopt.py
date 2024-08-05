import json
import jax
import jax.numpy as np
import numpy 
from jax import value_and_grad
import nlopt 
import sys
sys.path.append('iteration')
import fourier
pi = np.pi

with open('initfiles/init_args.json', 'r') as f:    # 传入地址
    args = json.load(f)
globals().update(args)

params = np.array([1, 2, 3]) 
print(params)
n = 0
# def loss(params, grad):
#     global n
#     n = n + 1     
#     print('input = ', n, type(params), params, type(grad), grad)
#     v = np.sum(pow(params, 2))
#     if grad.size > 0:
#         grad = 2*params
#     v = numpy.float64(v)
#     print('output = ', type(v), v, type(grad), grad)
#     return v
a = np.ones((5))
y = np.array([0])
# def lossg(params, y):
#     y = y.at[:].set(np.sum(params**2))
#     return y

def lossg(params):
    print(params)
    v = params[0]**2 + params[1]**2 + params[2]**2
    return v

loss_val, gradient = value_and_grad( lossg, allow_int = True)(params)   

# primals, f_vjp = jax.vjp(lambda params :lossg(params, y), y)
# print(primals, f_vjp)
# gradient = f_vjp(params)
# print(gradient)



# primals, f_vjp = jax.vjp(lossg, 1)
# print(primals, f_vjp)
# gradient = f_vjp(params)
# print(gradient)






# def objective_function_nlopt(params, grad):#=None
#     print('input = ', n, type(params), params, type(grad), grad)
#     # loss_val, gradient = loss(params)

#     loss_val = numpy.float64(loss_val) 
#     gradient = numpy.array(gradient)
#     if grad.size > 0:
#         grad = gradient
#     print('output = ', type(loss_val), loss_val, type(grad), grad)
#     return  loss_val

# opt = nlopt.opt(nlopt.LD_MMA, len(params))
# opt.set_min_objective(loss)
# opt.set_ftol_abs(1e-8)
# xopt = opt.optimize(params)
# print('n = ', n, 'loss = ', opt.last_optimum_value(), xopt)  



# def fx(x, grad = None):
#     print(x)
#     return 2*x**2-5*x
# x0 = [0]
# opt = nlopt.opt(nlopt.LN_SBPLX, 1)
# opt.set_min_objective(fx)
# opt.set_ftol_rel(1e-6)
# xopt = opt.optimize(x0)
# print(xopt)

# i = 0
# def obj(x, grad=None):
#     x1 = x[0]
#     x2 = x[1]
#     x3 = x[2]
#     ob = 6*x1+4*x2+3*x3
#     if x1+x2+x3>95 or 6*x1+5*x2+2*x3>400 or 5*x1+2*x2>200 or 12*x1+10*x2+16*x3>1200:
#         ob = 1
#     global i
#     i += 1
#     return 1/ob

# x1, x2, x3 = [1, 95], [1, 95], [1, 95]
# x0 = [15, 20, 12]
# # obj = Of()
### create object
# opt = nlopt.opt(nlopt.LD_LBFGS, 3)
### Objective function
# opt.set_min_objective(obj)
### Bound constraints
# opt.set_lower_bounds([x1[0], x2[0], x3[0]])
# opt.set_upper_bounds([x1[1], x2[1], x3[1]])
### Nonlinear constraints
# opt.add_inequality_constraint(fc, tol=0)
# opt.add_equality_constraint(h, tol=0)
# opt.add_inequality_mconstraint(c, tol)
# opt.add_equality_mconstraint(c, tol)
### Stopping criteria
# opt.set_stopval(stopval)
# opt.set_ftol_rel(tol)
# opt.set_ftol_abs(tol)
# opt.set_xtol_rel(1e-20)
# opt.set_xtol_abs(tol)
# opt.set_x_weights(w)
# opt.set_maxeval(100000)
# opt.set_maxtime(maxtime)
### Initial step size (derivative-free optimization)
# opt.set_initial_step([0.5, 0.5, 0.5])
### Pseudorandom numbers (stochastic optimization algorithms)
# nlopt.srand(seed) (seed is an integer)
# nlopt.srand_time()
### Performing the optimization
# x = opt.optimize(x0)
### result
# opt_val = opt.last_optimum_value()
# result = opt.last_optimize_result()
# print(x)
# print(1/(opt.last_optimum_value()))
# print(opt.last_optimize_result())
# print(i)

import math as m
def myfunc(x, grad):
    print('input',x, grad)
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 0.5 / m.sqrt(x[1])
    fx = m.sqrt(x[1])
    print('output',fx,grad)
    return fx

def myconstraint(x, grad, a, b):
    if grad.size > 0:
        grad[0] = 3 * a * (a*x[0] + b)**2
        grad[1] = -1.0
    # print(grad)
    return (a*x[0] + b)**3 - x[1]

opt = nlopt.opt(nlopt.LD_MMA, 2)
opt.set_lower_bounds([-float('inf'), 0])
opt.set_min_objective(myfunc)
opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad, 2, 0), 1e-8)
opt.add_inequality_constraint(lambda x, grad: myconstraint(x,grad, -1, 1), 1e-8)
opt.set_xtol_rel(1e-4)
x0 = [1.234, 5.678]
x = opt.optimize(x0)
minf = opt.last_optimum_value()
print('optimum at ', x)
print('minimum value = ', minf)
print('result code = ', opt.last_optimize_result())
print('nevals = ', opt.get_numevals())
print('initial step =', opt.get_initial_step(x0))


