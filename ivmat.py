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
  def dot(cls, x, y): # matrix operation
    if not len(x[0]) == len(y):
      raise cls.DimentionNotMatchError('dot dimention unmatch')
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
  
  def abs(self):
    mat = ivmat([[None for col in range(self.shape[1])] for row in range(self.shape[0])])
    for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        iv = self[i][j]
        mat[i][j] = max(map(lambda v: abs(v), iv[0]))
    return mat

  def get_max_width_and_index(self):
    """
    Returns:
      (index_of_max_width, max_width)
    """
    X = ivmat._flatten(self)
    # [(0, width(x_1)), (1, width(x_2)),...]
    X_width = list(enumerate(map(lambda x: x[0][1]-x[0][0] , X)))
    X_width.sort(key=lambda x: -x[1])#降順にソート
    return X_width[0]
  
  def max_width(self):
    index_of_max_width, max_width = self.get_max_width_and_index()
    return max_width
    
  def argmax_width(self):
    index_of_max_width, max_width = self.get_max_width_and_index()
    return index_of_max_width

  @property
  def norm(self):
    absmat = self.abs()
    return max([sum(row) for row in absmat])
    
  @classmethod
  def _flatten(cls, x): # 1-D
    return reduce(lambda a,b: a+b, x)
    
  def reshape(self, row_num, col_num):
    if not row_num * col_num == self.size:
      raise self.SizeNotMatchError('reshape method cannnot change elements size')    
    mat = ivmat([[None for col in range(col_num)] for row in range(row_num)])
    for i, x in enumerate(ivmat._flatten(self)):
      row_index = i // col_num
      col_index = i % col_num
      mat[row_index][col_index] = x
    return mat

  def copy(self):
    return self.reshape(*self.shape)
  
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
  def eye(cls, n):
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

  @classmethod
  def is_in(cls, A, B):
    """
    Parmas:
      A: ivmat obj shape == (n,1)
      A: ivmat obj shape == (n,1)
    Return:
      A in B
    """
    A = cls._flatten(A)
    B = cls._flatten(B)
    for flag in map(lambda (a,b): a in b, zip(A,B)):
      if not flag:
        return False
    else:
      return True

  @classmethod
  def is_0_in(cls, B):#全成分に0が含まれているか
    A = ivmat([[0] for i in range(B.shape[0])])
    return cls.is_in(A,B)

  @classmethod 
  def is_empty(cls, x):
    for iv in cls._flatten(x):
      if not iv == interval():
        return False
    return True


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
    self.f = f # fmat object
    self.f_grad = f_grad # fmat object
    self.X = X # ivmat object
    # self.y = self.X.midpoint.to_scalar()
    # self.Y = self.f_grad.apply_args(self.X).midpoint.to_scalar().get_pinv()
    # self.Z = self.X-self.y
    self.dim = len(self.f)
    self._NO_SOLUTIONS_FLAG = '_NO_SOLUTIONS_FLAG'
    self._EXACT_1_SOLUTION_FLAG = '_EXACT_1_SOLUTION_FLAG'
    self._MULTI_SOLUTIONS_FLAG = '_MULTI_SOLUTIONS_FLAG' # greater than 1 solution
    self._UNCLEAR_SOLUTION_FLAG = '_UNCLEAR_SOLUTION_FLAG'

  def refine(self, X_0, iter_num=10, trace=False):
    """
    X_0: initial box including unique solution.
    """
    # initialize
    X = X_0
    y = X.midpoint.to_scalar()
    Y = self.f_grad.apply_args(X).midpoint.to_scalar().get_pinv()
    Z = X - y
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
        Y = self.f_grad.apply_args(X).midpoint.to_scalar().get_pinv()
        Z = X - y
    else:
      return X

  def is_make_sure_solution_exist(self,X, trace=False):
    """
    解の存在を保証する判定法
    """
    Y = self.f_grad.apply_args(X).midpoint.to_scalar().get_pinv()
    R = ivmat.eye(self.dim) - ivmat.dot(Y, self.f_grad.apply_args(X))
    y = X.midpoint
    KX = y - ivmat.dot(Y, self.f.apply_args(y)) + ivmat.dot(R, (X - y))
    if trace:
      print 'R.norm == ', R.norm
      print 'ivmat.is_empty(KX & X) == ', ivmat.is_empty(KX & X)
      print 'KX == ', KX
      print 'X == ', X
      print 'ivmat.is_in(KX, X) == ', ivmat.is_in(KX, X)
    #step4
    if ivmat.is_empty(KX & X):
      return self._NO_SOLUTIONS_FLAG
    if ivmat.is_in(KX, X):
      if R.norm < 1:
        return self._EXACT_1_SOLUTION_FLAG
      else:
        return self._MULTI_SOLUTIONS_FLAG
    else:
      return self._UNCLEAR_SOLUTION_FLAG
      
  def is_make_sure_not_solution_exist(self, X, trace=False):
    """
    解の非存在を保証する判定法
    """    
    f = self.f
    FX = f.apply_args(X)
    if ivmat.is_0_in(FX): #全成分に0が含まれている
      return self._UNCLEAR_SOLUTION_FLAG
    else:
      return self._NO_SOLUTIONS_FLAG
    
  @classmethod
  def bisect(cls, X):
    if not X.shape[1] == 1:
      raise ivmat.DimentionNotMatchError()
    index = X.argmax_width()
    iv_inf,iv_sup = X[index][0][0][0],X[index][0][0][1]
    iv_mid = (iv_inf + iv_sup) / 2.0    
    X_1 = X.copy()    
    X_1[index][0] = interval[iv_inf, iv_mid]
    X_2 = X.copy()
    X_2[index][0] = interval[iv_mid, iv_sup]
    return X_1,X_2
    
  def find_all_solution(self, trace=False):
    # step1
    S = [self.X]
    T = []
    while(True):
      if trace:
        print '--------'*5
        pprint(S)
        pprint(T)
        print '----@@@----'*5
      # step2
      if not S: # S is empty
        break
      X = S.pop(0)
      #step3
      flag = self.is_make_sure_not_solution_exist(X, trace)
      if flag == self._NO_SOLUTIONS_FLAG:
        continue # to step2
      else:
        #step4
        flag = self.is_make_sure_solution_exist(X, trace)
        if flag == self._NO_SOLUTIONS_FLAG:#解が存在しないことが確定
          continue # to step2
        #step5
        elif flag == self._UNCLEAR_SOLUTION_FLAG:#解の存在・非存在について何もわからない
          X_1,X_2 = Krawczyk.bisect(X)
          S.append(X_1)
          S.append(X_2)
          continue # to step2
        #step6,7
        elif flag == self._EXACT_1_SOLUTION_FLAG:# 解が1つのみ
          #step6
          T.append(X)
          continue # to step2
        elif flag == self._MULTI_SOLUTIONS_FLAG:# 解が複数
          #step7
          X_1,X_2 = Krawczyk.bisect(X)
          S.append(X_1)
          S.append(X_2)
          continue # to step2
    # Tは解が一意に存在するboxのlist
    return map(lambda x: self.refine(x), T)


