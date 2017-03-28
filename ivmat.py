# -*- coding: utf-8 -*-
#!/usr/bin/env python

from interval import interval, inf
import numpy as np
from pprint import pprint
from _logger import logger


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

    @property
    def midpoint(self):
        mat = ivmat([[None for col in range(self.shape[1])]
                     for row in range(self.shape[0])])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i][j] = self[i][j].midpoint
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

    def get_pinv(self):
        np_pinv = np.around(np.linalg.pinv(np.array(self)), decimals=2)
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


class fmat(list):

    def apply_args(self, X_mat):
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

    def __init__(self, f, f_grad, X):
        self.f = f  # fmat object
        self.f_grad = f_grad  # fmat object
        self.X = X  # ivmat object
        # self.y = self.X.midpoint.to_scalar()
        # self.Y = self.f_grad.apply_args(self.X).midpoint.to_scalar().get_pinv()
        # self.Z = self.X-self.y
        self.dim = len(self.f)
        self._NO_SOLUTIONS_FLAG = '_NO_SOLUTIONS_FLAG'
        self._EXACT_1_SOLUTION_FLAG = '_EXACT_1_SOLUTION_FLAG'
        self._MULTI_SOLUTIONS_FLAG = '_MULTI_SOLUTIONS_FLAG'  # greater than 1 solution
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
        if trace:
            pprint(X)
        for i in range(iter_num):
            left = y - ivmat.dot(Y, self.f.apply_args(y))
            right = ivmat.dot(ivmat.eye(self.dim) -
                              ivmat.dot(Y, self.f_grad.apply_args(X)), Z)
            KX = left + right
            X = KX & X
            if trace:
                print '---------', i, '------------------'
                pprint(X)
            if ivmat.is_empty(X):
                print '---------- ivmat.is_empty(X) == True -----'+'---'*30
                pprint(X)
                break  # return X
            # update
            y = X.midpoint
            Y = self.f_grad.apply_args(X).midpoint.to_scalar().get_pinv()
            Z = X - y
        return X

    def get_R_and_KX(self, X):
        F1_X = self.f_grad.apply_args(X)  # F'(X)
        mF1_X = F1_X.midpoint.to_scalar()  # m(F'(X))
        Y = mF1_X.get_pinv()
        R = ivmat.eye(self.dim) - ivmat.dot(Y, F1_X)
        y = X.midpoint
        KX = y - ivmat.dot(Y, self.f.apply_args(y)) + ivmat.dot(R, (X - y))
        return R, KX

    def is_make_sure_solution_exist(self, X, trace=False):
        """
        解の存在を保証する判定法
        """
        mf_grad_x = self.f_grad.apply_args(X).midpoint.to_scalar()  # m(f'(x))
        R, KX = self.get_R_and_KX(X)
        logger.info((
            '\n'
            'X: {}\n'
            'KX: {}\n'
            'R.norm: {}\n'
            'm(f\'(x)): {}\n'
            'is_empty(KX & X):{}\n'
            'is_in(KX, X):{}\n').format(X, KX, R.norm, mf_grad_x, ivmat.is_empty(KX & X), ivmat.is_in(KX, X)))

        # step4
        new_X = KX & X
        if ivmat.is_empty(new_X):
            return new_X, self._NO_SOLUTIONS_FLAG
        if ivmat.is_in(KX, X):
            if R.norm < 1:
                return new_X, self._EXACT_1_SOLUTION_FLAG
            else:
                return new_X, self._MULTI_SOLUTIONS_FLAG
        else:
            return new_X, self._UNCLEAR_SOLUTION_FLAG

    def is_make_sure_not_solution_exist(self, X, trace=False):
        """
        解の非存在を保証する判定法
        """
        f = self.f
        FX = f.apply_args(X)
        if ivmat.is_0_in(FX):  # 全成分に0が含まれている
            return self._UNCLEAR_SOLUTION_FLAG
        else:
            return self._NO_SOLUTIONS_FLAG

    @classmethod
    def bisect(cls, X, trace):
        if not X.shape[1] == 1:
            raise ivmat.DimentionNotMatchError()
        index = X.argmax_width()
        iv_inf, iv_sup = X[index][0][0][0], X[index][0][0][1]
        iv_mid = (iv_inf + iv_sup) / 2.0
        X_1 = X.copy()
        X_1[index][0] = interval[iv_inf, iv_mid]
        X_2 = X.copy()
        X_2[index][0] = interval[iv_mid, iv_sup]
        return X_1, X_2

    def prove_algorithm(self, X, init_X, max_iter_num=20, trace=False):
        """
        少しシフトすれば解を観測できる区間に対して検証する
        True or Falseを返すべき
        唯一の解が存在すればTrueなのでその先でTに追加
        それ以外は、splitする
        """
        MU = 0.9
        TAU = 1.01
        d = inf
        d_prev = inf
        k = 0
        X_0 = X  # 最初のX
        while(True):
            if not((d < MU * d_prev or (d == inf and d_prev == inf)) and ivmat.is_in(X, init_X) and k < max_iter_num):
                break
            R, new_X = self.get_R_and_KX(X)
            kx_and_x, flag = self.is_make_sure_solution_exist(X)
            if flag == self._EXACT_1_SOLUTION_FLAG:
                return kx_and_x, flag
            if flag == self._NO_SOLUTIONS_FLAG:
                return kx_and_x, flag  # new_X.is_empty() == True
            d_prev = d
            d = ivmat.hausdorff_distance(X, new_X)
            k += 1
            X = new_X.midpoint + TAU * (new_X - new_X.midpoint)
        return X_0, self._UNCLEAR_SOLUTION_FLAG

    def find_all_solution(self, trace=False, cnt_max=1000):
        init_X = self.X
        # step1
        S = [self.X]
        T = []
        U = []  # これ以上は浮動小数点演算の限界
        logger.info('[step 1] init_X:{}, len(S):{}, len(T):{}, len(U)'.format(init_X, len(S), len(T), len(U)))
        cnt = 0
        prove_trace_flag = False
        S_sizes = []
        while(True):
            cnt += 1
            S_sizes.append(len(S))
            if cnt > cnt_max:
                break

            logger.info('cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))
            if cnt % 50 == 0:
                logger.info('\n--- S ---\n{}'.format(S[:10]))
                logger.info('\n--- T ---\n{}'.format(T[:10]))
                logger.info('---' * 20)
            # step2
            logger.info('[step 2]')
            if not S:  # S is empty
                break
            X = S.pop(0)
            logger.info('X:{}'.format(X))

            if X.max_width() < 1e-7:
                # 限界の精度を決める
                U.append(X)
                logger.info('limit width X.max_width():{}'.format(X.max_width()))
                continue  # to step2

            # step3
            flag = self.is_make_sure_not_solution_exist(X, trace)
            logger.info('[step 3] X:{}, 0-solution:{}'.format(X, flag))
            if flag == self._NO_SOLUTIONS_FLAG:
                logger.info('[step 3] to [step 2]')
                continue  # to step2
            else:
                # step4
                X, flag = self.is_make_sure_solution_exist(X, trace)
                logger.info('[step 4] X:{}, flag:{}'.format(X, flag))
                if flag == self._NO_SOLUTIONS_FLAG:  # 解が存在しないことが確定
                    logger.info('[step 4] to [step 2]')
                    continue  # to step2
                # step5
                elif flag == self._UNCLEAR_SOLUTION_FLAG:  # 解の存在・非存在について何もわからない
                    X, prove_flag = self.prove_algorithm(X, init_X, trace=prove_trace_flag)
                    logger.info('[step 5] X:{}, prove_flag:{}'.format(X, prove_flag))
                    if prove_flag == self._NO_SOLUTIONS_FLAG:
                        logger.info('[step 5] to [step 2]')
                        continue  # to step2
                    elif prove_flag == self._UNCLEAR_SOLUTION_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 5] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 5] to [step 2]')
                        continue  # to step2
                    elif prove_flag == self._EXACT_1_SOLUTION_FLAG:
                        T.append(X)
                        continue
                    elif prove_flag == self._MULTI_SOLUTIONS_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 5] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 5] to [step 2]')
                        continue  # to step2
                    else:
                        print prove_flag
                        print '[step5] なんか変'

                # step6,7
                elif flag == self._EXACT_1_SOLUTION_FLAG:
                    # step6
                    T.append(X)
                    logger.info('[step 6] exact 1 solution in X:{}'.format(X))
                    logger.info('[step 6] to [step 2]')
                    continue  # to step2
                elif flag == self._MULTI_SOLUTIONS_FLAG:  # 解が複数
                    # step7
                    X, prove_flag = self.prove_algorithm(X, init_X, trace=prove_trace_flag)
                    logger.info('[step 7] X:{}, prove_flag:{}'.format(X, flag))

                    if prove_flag == self._NO_SOLUTIONS_FLAG:
                        logger.info('[step 7] to [step 2]')
                        continue
                    elif prove_flag == self._UNCLEAR_SOLUTION_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 7] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 7] to [step 2]')
                        continue  # to step2
                    elif prove_flag == self._EXACT_1_SOLUTION_FLAG:
                        T.append(X)
                        logger.info('[step 7] exact 1 solution in X:{}'.format(X))
                        logger.info('[step 7] to [step 2]')
                        continue
                    elif prove_flag == self._MULTI_SOLUTIONS_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 7] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 7] to [step 2]')
                        continue  # to step2
                    else:
                        print prove_flag
                        print '[step7] なんか変'

        # Tは解が一意に存在するboxのlist
        logger.info('Loop end. cnt:{}, len(S):{}, len(T):{}'.format(cnt, len(S), len(T)))
        print
        print cnt
        print('---------- 最終的なS[:10] -----------')
        pprint(S[:10])
        print('---------- 最終的なU[:10] -----------')
        pprint(U[:10])
        print('---------- 最終的なT -----------')
        pprint(T)

        return map(lambda x: self.refine(x), T), S_sizes
