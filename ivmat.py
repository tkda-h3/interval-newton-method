# -*- coding: utf-8 -*-
#!/usr/bin/env python

from interval import interval
import numpy as np


class ivmat(list):

    class DimentionNotMatchError(Exception):
        pass

    class NotListError(Exception):
        pass

    class SizeNotMatchError(Exception):
        pass

    class NotMidpointError(Exception):
        pass

    class EmptyIntervalError(Exception):
        pass

    def _get_shape(self, X):
        if isinstance(X, list):
            for index, x in enumerate(X):
                if index == 0 and isinstance(x, list):
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
        self.size = reduce(lambda x, y: x * y, self.shape)
        # self.midpoint = self._get_midpoint()
        super(ivmat, self).__init__(args)

    def __pos__(self):  # +x
        return self

    def __neg__(self):  # -x
        return ivmat(map(lambda x_row:
                         [-x for x in x_row],
                         self))

    def __add__(self, other):  # x + y
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x + y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __radd__(self, other):  # y + x
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x + y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __sub__(self, other):  # x - y
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x - y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __rsub__(self, other):  # x - y
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [y - x for y, x in zip(y_row, x_row)],
                         zip(other, self)))

    def __mul__(self, other):  # x * y
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x * y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __rmul__(self, other):  # y * x
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x * y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __truediv__(self, other):  # self / other
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x / y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    def __rtruediv__(self, other):  # other / self
        if isinstance(other, int) or isinstance(other, float):
            other = ivmat.uniform_mat(other, self.shape)
        return ivmat(map(lambda (x_row, y_row):
                         [x / y for x, y in zip(x_row, y_row)],
                         zip(other, self)))

    def __and__(self, other):
        return ivmat(map(lambda (x_row, y_row):
                         [x & y for x, y in zip(x_row, y_row)],
                         zip(self, other)))

    @classmethod
    def dot(cls, x, y):  # matrix operation
        if not len(x[0]) == len(y):
            raise cls.DimentionNotMatchError('dot dimention unmatch')
        mat = ivmat([[None for col in range(len(y[0]))]
                     for row in range(len(x))])
        for i in range(len(x)):
            for j in range(len(y[0])):
                sum = 0
                for k in range(len(x[0])):
                    sum += x[i][k] * y[k][j]
                mat[i][j] = sum
        return mat

    @classmethod
    def mid(cls, x, scalar=True):
        """
        get the midpoint of x

        Param:
          scalar: 
            if scalar is True, element of matrix is scalar
        """
        mat = ivmat([[None for col in range(x.shape[1])]
                     for row in range(x.shape[0])])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i][j] = x[i][j].midpoint

        if scalar:
            mat = mat.to_scalar()
        return mat

    def abs(self):
        mat = ivmat([[None for col in range(self.shape[1])]
                     for row in range(self.shape[0])])
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
        X_width = list(enumerate(map(lambda x: x[0][1] - x[0][0], X)))
        X_width.sort(key=lambda x: -x[1])  # 降順にソート
        return X_width[0]

    def max_width(self):
        _, max_width = self.get_max_width_and_index()
        return max_width

    def argmax_width(self):
        index_of_max_width, max_width = self.get_max_width_and_index()
        return index_of_max_width

    @property
    def norm(self):
        absmat = self.abs()
        return max([sum(row) for row in absmat])

    @classmethod
    def hausdorff_distance(cls, X, Y):
        flat_X = cls._flatten(X)
        flat_Y = cls._flatten(Y)
        dist_vec = map(lambda (x, y): max(abs(x[0][0] - y[0][0]), abs(x[0][1] - y[0][1])), zip(flat_X, flat_Y))
        return max(dist_vec)

    @classmethod
    def _flatten(cls, x):  # 1-D
        return reduce(lambda a, b: a + b, x)

    def reshape(self, row_num, col_num):
        if not row_num * col_num == self.size:
            raise self.SizeNotMatchError(
                'reshape method cannnot change elements size')
        mat = ivmat([[None for col in range(col_num)]
                     for row in range(row_num)])
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
                if iv == interval():
                    raise self.EmptyIntervalError()
                if not iv[0][0] == iv[0][1]:
                    return False
        return True

    def to_scalar(self):
        mat = ivmat([[None for col in range(self.shape[1])]
                     for row in range(self.shape[0])])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                iv = self[i][j]
                if not self.is_midpoint():
                    raise self.NotMidpointError()
                mat[i][j] = iv[0][0]
        return mat

    def to_interval(self):
        mat = ivmat([[None for col in range(self.shape[1])]
                     for row in range(self.shape[0])])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i][j] = interval.cast(self[i][j])
        return mat


    @classmethod
    def pinv(cls, x):
        np_pinv = np.around(np.linalg.pinv(np.array(x)), decimals=5)
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
        for flag in map(lambda (a, b): a in b, zip(A, B)):
            if not flag:
                return False
        else:
            return True

    @classmethod
    def is_0_in(cls, B):  # 全成分に0が含まれているか
        A = ivmat([[0] for i in range(B.shape[0])])
        return cls.is_in(A, B)

    @classmethod
    def is_empty(cls, x):
        """
        1つでも interval()ならTrue
        """
        for iv in cls._flatten(x):
            if iv == interval():
                return True
        return False

    @classmethod
    def uniform_mat(cls, value, shape):
        size = shape[0] * shape[1]
        if not isinstance(size, int):
            raise cls.NotIntError()
        args = [[value for j in range(shape[1])] for i in range(shape[0])]
        return ivmat(args)

    def extend_width(self, expantion_ratio=1.01):
        return self.midpoint + (expantion_ratio * (self - self.midpoint))

    @classmethod
    def max(cls, x):
        """
        xがscalarの時、要素の最大値を返す
        """
        return max(cls._flatten(x))

    
