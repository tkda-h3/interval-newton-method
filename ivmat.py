# -*- coding: utf-8 -*-
#!/usr/bin/env python

from interval import interval
import numpy as np
from pprint import pprint
  
class ivmat(list):
  class DimentionNotMatchError(Exception):
    pass

  class NotListError(Exception):
    pass

  class SizeNotMatchError(Exception):
    pass

  class NotMidpointError(Exception):
    pass

  def _get_shape(self, X):
    if isinstance(X, list):            
      for index, x in enumerate(X):
        if index==0 and isinstance(x, list):
          col_num = len(x)
        elif isinstance(x, list):
          if not col_num == len(x):
            raise self.DimentionNotMatchError()
        else:
          raise self.NotListError()
      return (len(X), col_num)
    else:
      raise self.NotListError()

  def __init__(self, args):
    self.shape = self._get_shape(args)
    self.size = reduce(lambda x,y: x*y, self.shape)
    #self.midpoint = self._get_midpoint()
    super(ivmat, self).__init__(args)

  def __pos__(self): # +x
    return self

  def __neg__(self): # -x
    return ivmat(map(lambda x_row: 
                     [-x for x in x_row], 
                     self))

  def __add__(self, other): # x + y
    return ivmat(map(lambda (x_row,y_row): 
                     [x+y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))

  def __radd__(self, other): # y + x
    return ivmat(map(lambda (x_row,y_row): 
                     [x+y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))

  def __sub__(self, other): # x - y
    return ivmat(map(lambda (x_row,y_row): 
                     [x-y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))

  def __mul__(self, other): # x * y
    return ivmat(map(lambda (x_row,y_row): 
                     [x*y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))

  def __rmul__(self, other): # y * x
    return ivmat(map(lambda (x_row,y_row): 
                     [x*y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))

  def __and__(self, other):
    return ivmat(map(lambda (x_row,y_row): 
                     [x&y for x,y in zip(x_row, y_row)], 
                     zip(self, other)))


  @classmethod
  def dot(self, x, y): # matrix operation
    if not len(x[0]) == len(y):
      raise self.DimentionNotMatchError('dot dimention unmatch')
    mat = ivmat([[None for col in range(len(y[0]))] for row in range(len(x))])
    for i in range(len(x)):
      for j in range(len(y[0])):
        sum = 0
        for k in range(len(x[0])):
          sum += x[i][k] * y[k][j]
        mat[i][j] = sum
    return mat

  @property
  def midpoint(self):
    mat = ivmat([[None for col in range(self.shape[1])] for row in range(self.shape[0])])
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        mat[i][j] = self[i][j].midpoint
    return mat
    
  def _flatten(self): # 1-D
    return reduce(lambda x,y: x+y, self)
    
  def reshape(self, row_num, col_num):
    if not row_num * col_num == self.size:
      raise self.SizeNotMatchError('reshape method cannnot change elements size')    
    mat = ivmat([[None for col in range(col_num)] for row in range(row_num)])
    for i, x in enumerate(self._flatten()):
      row_index = i // col_num
      col_index = i % col_num
      mat[row_index][col_index] = x
    return mat
    
  def is_midpoint(self):
    for i in range(self.shape[0]):
      for j in range(self.shape[1]):
        iv = self[i][j]
        if not iv[0][0] == iv[0][1]:
          return False
    return True
    
  def to_scalar(self):
    mat = ivmat([[None for col in range(self.shape[1])] for row in range(self.shape[0])])        
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        iv = self[i][j]
        if not self.is_midpoint(): 
          raise self.NotMidpointError()
        mat[i][j] = iv[0][0]
    return mat
    
  def to_interval(self):
    mat = ivmat([[None for col in range(self.shape[1])] for row in range(self.shape[0])])        
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        mat[i][j] = interval.cast(self[i][j])
    return mat
    
  def get_pinv(self):
    np_pinv =  np.around(np.linalg.pinv(np.array(self)), decimals=2)
    return ivmat(np_pinv.tolist()).to_interval()
    
  @classmethod
  def eye(self, n):
      """
      Returns:
          ans = ivmat(
              [1,0,...,0],
              [0,1,...,0],
                      :
              [0,...,0,1]
          )
          ans.shape == (n,n)
      """
      return ivmat(np.eye(n).tolist())


class fmat(list):
  def apply_args(self,X_mat):
    """
    Params:
        X_mat: ivmat([[x_1], [x_2]])
    Returns:
        ivmat
    """
    X = map(lambda x: x[0], X_mat)        
    ans = []
    for f_row in self:
      tmp = []
      for f in f_row:
        tmp.append(f(*X))
      ans.append(tmp)
    return ivmat(ans)            


class Krawczyk():
  def __init__(self, f,f_grad, X):
    self.f = f
    self.f_grad = f_grad
    self.X = X
    self.y = self.X.midpoint.to_scalar()
    self.Y = self.f_grad.apply_args(self.X).midpoint.to_scalar().get_pinv()
    self.Z = self.X-self.y
    self.dim = len(self.f)

  def run(self, iter_num=10, trace=False):
    # initialize
    X = self.X
    y = X.midpoint
    Y = self.Y
    Z = self.Z
    if trace: pprint(X)
    for i in range(iter_num):
        left = y - ivmat.dot(Y, self.f.apply_args(y))
        right = ivmat.dot(ivmat.eye(self.dim) - ivmat.dot(Y, self.f_grad.apply_args(X)), Z)
        KX = left + right
        X = KX & X
        if trace:
          print '---------', i,'------------------'
          pprint(X)        
        #update
        y = X.midpoint
        Y =  self.f_grad.apply_args(X).midpoint.to_scalar().get_pinv()
        Z = X - y
