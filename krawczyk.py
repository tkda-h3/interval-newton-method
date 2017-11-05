# -*- coding: utf-8 -*-
#!/usr/bin/env python

from interval import interval, inf
import numpy as np
from pprint import pprint
from _logger import logger

from ivmat import ivmat as ip
from consts import(
    _NO_SOLUTIONS_FLAG,
    _NO_MINMUM_FLAG,
    _EXACT_1_SOLUTION_FLAG,
    _MULTI_SOLUTIONS_FLAG,
    _UNCLEAR_SOLUTION_FLAG,
)


class Krawczyk():

    def __init__(self, f, df, X_init):
        """
        f: f=0を満たすxを探す
        df: fの微分
        X_init: 探索box
        """
        self.f = f  # fmat object
        self.df = df  # fmat object
        self.X_init = X_init  # ivmat object
        self.dim = len(self.f)
        
    def refine(self, X_0, iter_num=10, trace=False):
        """
        X_0: initial box including unique solution.
        """
        # initialize
        X = X_0
        y = ip.mid(X)
        Y = ip.pinv(ip.mid(self.df(X)))
        Z = X - y
        for i in range(iter_num):
            left = y - ip.dot(Y, self.f(y))
            right = ip.dot(ip.eye(self.dim) -
                              ip.dot(Y, self.df(X)), Z)
            KX = left + right
            X = KX & X
            if ip.is_empty(X):
                break  # return X
            # update
            y = ip.mid(X)
            Y = ip.pinv(ip.mid(self.df(X)))
            Z = X - y
        return X

    def get_R_and_KX(self, X):
        dfx = self.df(X)  # DF'(X)
        mdfx = ip.mid(dfx)  # m(DF'(X))
        logger.info('m(F\'(x)): {}'.format(mdfx))
        logger.info('ip.max(mdfx): {}'.format(ip.max(mdfx.to_interval().abs())))
        digit = int(np.log10(ip.max(mdfx.to_interval().abs()))) + 20
        scale = 10.0 ** digit
        logger.info('scale: {}'.format(scale))
        if -100 < digit < 100:  # 値が正常範囲内なら工夫なし
            Y = ip.pinv(mdfx)
            R = ip.eye(self.dim) - ip.dot(Y, dfx)
            logger.info('[no scale] Y:{}'.format(Y))
        else:  # オーバーフローが発生しないように工夫
            Y = ip.pinv(mdfx.__truediv__(scale))
            R = ip.eye(self.dim) - ip.dot(Y, dfx) * scale
            logger.info('[scale used] Y:{}'.format(Y))

        y = ip.mid(X)
        KX = y - ip.dot(Y, self.f(y)) + ip.dot(R, (X - y))
        return R, KX

    def is_make_sure_solution_exist(self, X, trace=False):
        """
        解の存在を保証する判定法
        """
        mdfx = ip.mid(self.df(X))  # m(f'(x))
        R, KX = self.get_R_and_KX(X)
        logger.info((
            '\n'
            'X: {}\n'
            'KX: {}\n'
            'R.norm: {}\n'
            'm(df(x)): {}\n'
            'is_empty(KX & X):{}\n'
            'is_in(KX, X):{}\n').format(X, KX, R.norm, mdfx, ip.is_empty(KX & X), ip.is_in(KX, X)))

        # step4
        new_X = KX & X
        if ip.is_empty(new_X):
            return new_X, _NO_SOLUTIONS_FLAG
        if ip.is_in(KX, X):
            if R.norm < 1:
                return new_X, _EXACT_1_SOLUTION_FLAG
            else:
                return new_X, _MULTI_SOLUTIONS_FLAG
        else:
            return new_X, _UNCLEAR_SOLUTION_FLAG

    def is_make_sure_not_solution_exist(self, X, trace=False):
        """
        解の非存在を保証する判定法
        """
        f = self.f
        FX = f(X)
        if ip.is_0_in(FX):  # 全成分に0が含まれている
            return _UNCLEAR_SOLUTION_FLAG
        else:
            return _NO_SOLUTIONS_FLAG

    @classmethod
    def bisect(cls, X, trace):
        if not X.shape[1] == 1:
            raise ip.DimentionNotMatchError()
        index = X.argmax_width()
        iv_inf, iv_sup = X[index][0][0][0], X[index][0][0][1]
        iv_mid = (iv_inf + iv_sup) / 2.0
        X_1 = X.copy()
        X_1[index][0] = interval[iv_inf, iv_mid]
        X_2 = X.copy()
        X_2[index][0] = interval[iv_mid, iv_sup]
        return X_1, X_2

    def prove_algorithm(self, X, init_X, max_iter_num=10, trace=False):
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
            if not((d < MU * d_prev or (d == inf and d_prev == inf)) and ip.is_in(X, init_X) and k < max_iter_num):
                break
            R, new_X = self.get_R_and_KX(X)
            kx_and_x, flag = self.is_make_sure_solution_exist(X)
            if flag == _EXACT_1_SOLUTION_FLAG:
                return kx_and_x, flag
            if flag == _NO_SOLUTIONS_FLAG:
                return kx_and_x, flag  # new_X.is_empty() == True
            d_prev = d
            d = ip.hausdorff_distance(X, new_X)
            k += 1
            X = ip.mid(new_X) + TAU * (new_X - ip.mid(new_X))
        return X_0, _UNCLEAR_SOLUTION_FLAG

    def find_all_solution(self, trace=False, cnt_max=1000, max_width=1e-8):
        init_X = self.X_init
        # step1
        S = [self.X_init]
        T = []
        U = []  # これ以上は浮動小数点演算の限界
        animation_box = []  # アニメーション作成のために過程を保存
        logger.info('[step 1] init_X:{}, len(S):{}, len(T):{}, len(U)'.format(init_X, len(S), len(T), len(U)))
        cnt = 0
        prove_trace_flag = False
        S_sizes = []
        T_sizes = []
        U_sizes = []
        animation_box.append([(S[0], _UNCLEAR_SOLUTION_FLAG)])
        while(True):
            cnt += 1
            S_sizes.append(len(S))
            T_sizes.append(len(T))
            U_sizes.append(len(U))
            if cnt > cnt_max:
                break

            logger.info('cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))

            # step2
            logger.info('[step 2]')
            if not S:  # S is empty
                break
            step2_X = X = S.pop(0)
            logger.info('[step 2] S pop X. X:{}'.format(X))

            if X.max_width() < max_width:
                # 限界の精度を決める
                U.append(X)
                logger.info('limit width X.max_width():{}'.format(X.max_width()))
                animation_box.append([(X, _UNCLEAR_SOLUTION_FLAG)])
                continue  # to step2

            # step3
            flag = self.is_make_sure_not_solution_exist(X, trace)
            logger.info('[step 3] X:{}, flag:{}'.format(X, flag))
            if flag == _NO_SOLUTIONS_FLAG:
                logger.info('[step 3] to [step 2]')
                animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                continue  # to step2
            else:
                # step4
                X, flag = self.is_make_sure_solution_exist(X, trace)
                logger.info('[step 4] X:{}, flag:{}'.format(X, flag))
                if flag == _NO_SOLUTIONS_FLAG:  # 解が存在しないことが確定
                    logger.info('[step 4] to [step 2]')
                    animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                    continue  # to step2
                
                # step5
                elif flag == _UNCLEAR_SOLUTION_FLAG or \
                     flag == _MULTI_SOLUTIONS_FLAG: # 解の存在・非存在判定が失敗した場合
                    X, prove_flag = self.prove_algorithm(X, init_X, trace=prove_trace_flag)
                    logger.info('[step 5] X:{}, prove_flag:{}'.format(X, prove_flag))
                    
                    if prove_flag == _NO_SOLUTIONS_FLAG:
                        logger.info('[step 5] to [step 2]')
                        animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                        continue  # to step2
                    elif prove_flag == _UNCLEAR_SOLUTION_FLAG or \
                         prove_flag == _MULTI_SOLUTIONS_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 5] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 5] to [step 2]')
                        animation_box.append([
                            (step2_X, _NO_SOLUTIONS_FLAG),
                            (X_1, _UNCLEAR_SOLUTION_FLAG),
                            (X_2, _UNCLEAR_SOLUTION_FLAG),
                        ])
                        continue  # to step2
                    elif prove_flag == _EXACT_1_SOLUTION_FLAG:
                        T.append(X)
                        animation_box.append([
                            (step2_X, _NO_SOLUTIONS_FLAG),
                            (X, _EXACT_1_SOLUTION_FLAG),
                        ])
                        continue
                    else:
                        logger.error('[step5] なんか変. prove_flag: {}'.format(prove_flag))
                        raise ''
                    
                # step6,7
                elif flag == _EXACT_1_SOLUTION_FLAG:
                    # step6
                    T.append(X)
                    logger.info('[step 6] exact 1 solution in X:{}'.format(X))
                    logger.info('[step 6] to [step 2]')
                    animation_box.append([
                        (step2_X, _NO_SOLUTIONS_FLAG),
                        (X, _EXACT_1_SOLUTION_FLAG),
                    ])
                    continue  # to step2

        # Tは解が一意に存在するboxのlist
        logger.info('Loop end. cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))
        print('Loop end. cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))
        print
        print cnt
        print('---------- 最終的なS[:10] -----------')
        pprint(S[:10])
        print('---------- 最終的なU[:10] -----------')
        pprint(U[:10])
        print('---------- 最終的なT -----------')
        pprint(T)

        return map(lambda x: self.refine(x), T), S_sizes, T_sizes, U_sizes, animation_box


    def find_global_minimum(self, f, tmp_min_sup=inf, trace=False, cnt_max=1000, max_width=1e-8):
        """
        Params:
          f: 最小化したい関数
          tmp_min_sup: 最小値の上限値（事前にnelder meadなどで求めた局所最適値を使うと良い）
        """
        class TmpMin():
            def __init__(self, tmp_min_sup):
                self.sup = tmp_min_sup

                
        init_X = self.X_init
        # step1
        S = [self.X_init]
        T = []
        U = []  # これ以上は浮動小数点演算の限界
        animation_box = []  # アニメーション作成のために過程を保存
        tmp_min = TmpMin(tmp_min_sup)
        logger.info('[step 1] init_X:{}, len(S):{}, len(T):{}, len(U)'.format(init_X, len(S), len(T), len(U)))

        cnt = 0
        prove_trace_flag = False
        S_sizes = []
        T_sizes = []
        U_sizes = []
        animation_box.append([(S[0], _UNCLEAR_SOLUTION_FLAG)])
        while(True):
            cnt += 1
            S_sizes.append(len(S))
            T_sizes.append(len(T))
            U_sizes.append(len(U))
            if cnt > cnt_max:
                break

            logger.info('cnt:{}, tmp_min.sup:{}, len(S):{}, len(T):{}, len(U):{}'\
                        .format(cnt, tmp_min.sup, len(S), len(T), len(U)))

            # step2
            logger.info('[step 2]')
            if not S:  # S is empty
                break
            step2_X = X = S.pop(0)
            logger.info('[step 2] S pop X. X:{}'.format(X))

            if X.max_width() < max_width:
                # 限界の精度を決める
                U.append(X)
                logger.info('limit width X.max_width():{}'.format(X.max_width()))
                animation_box.append([(X, _UNCLEAR_SOLUTION_FLAG)])
                continue  # to step2

            if f(X)[0][0][0].inf > tmp_min.sup: # 最小になり得ない
                animation_box.append([(step2_X, _NO_MINMUM_FLAG)])                
                logger.info('Not global minima. X: {}, f(X)[0][0][0].inf: {}, tmp_min.sup: {}'.format(X, f(X)[0][0][0].inf, tmp_min.sup))
                continue
            if f(X)[0][0][0].sup < tmp_min.sup: # 最小値の上限を更新
                tmp_min.sup = f(X)[0][0][0].sup
                logger.info('tmp_min.sup is updated to {}. X: {}, f(X)[0][0]: {}'.format(f(X)[0][0][0].sup, X, f(X)[0][0]))            
            
            # step3
            flag = self.is_make_sure_not_solution_exist(X, trace)
            logger.info('[step 3] X:{}, flag:{}'.format(X, flag))
            if flag == _NO_SOLUTIONS_FLAG:
                logger.info('[step 3] to [step 2]')
                animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                continue  # to step2
            else:
                # step4
                X, flag = self.is_make_sure_solution_exist(X, trace)
                logger.info('[step 4] X:{}, flag:{}'.format(X, flag))
                if flag == _NO_SOLUTIONS_FLAG:  # 解が存在しないことが確定
                    logger.info('[step 4] to [step 2]')
                    animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                    continue  # to step2
                
                # step5
                elif flag == _UNCLEAR_SOLUTION_FLAG or \
                     flag == _MULTI_SOLUTIONS_FLAG: # 解の存在・非存在判定が失敗した場合
                    X, prove_flag = self.prove_algorithm(X, init_X, trace=prove_trace_flag)
                    logger.info('[step 5] X:{}, prove_flag:{}'.format(X, prove_flag))
                    
                    if prove_flag == _NO_SOLUTIONS_FLAG:
                        logger.info('[step 5] to [step 2]')
                        animation_box.append([(step2_X, _NO_SOLUTIONS_FLAG)])
                        continue  # to step2
                    elif prove_flag == _UNCLEAR_SOLUTION_FLAG or \
                         prove_flag == _MULTI_SOLUTIONS_FLAG:
                        X_1, X_2 = Krawczyk.bisect(X, trace)
                        S.append(X_1)
                        S.append(X_2)
                        logger.info('[step 5] bisect is succeeded\n--- X_1 ---\n{} \n--- X_2 ---\n{}'.format(X_1, X_2))
                        logger.info('[step 5] to [step 2]')
                        animation_box.append([
                            (step2_X, _NO_SOLUTIONS_FLAG),
                            (X_1, _UNCLEAR_SOLUTION_FLAG),
                            (X_2, _UNCLEAR_SOLUTION_FLAG),
                        ])
                        continue  # to step2
                    elif prove_flag == _EXACT_1_SOLUTION_FLAG:
                        if f(X)[0][0][0].sup < tmp_min.sup: # 最小値の上限を更新
                            tmp_min.sup = f(X)[0][0][0].sup
                            logger.info('tmp_min.sup is updated to {}. X: {}, f(X)[0][0]: {}'.format(f(X)[0][0][0].sup, X, f(X)[0][0]))
                            
                        T.append(X)                            
                        animation_box.append([
                            (step2_X, _NO_SOLUTIONS_FLAG),
                            (X, _EXACT_1_SOLUTION_FLAG),
                        ])
                        continue
                    else:
                        logger.error('[step5] なんか変. prove_flag: {}'.format(prove_flag))
                        raise ''
                    
                # step6,7
                elif flag == _EXACT_1_SOLUTION_FLAG:
                    if f(X)[0][0][0].sup < tmp_min.sup: # 最小値の上限を更新
                        tmp_min.sup = f(X)[0][0][0].sup
                        logger.info('tmp_min.sup is updated to {}. X: {}, f(X)[0][0]: {}'.format(f(X)[0][0][0].sup, X, f(X)[0][0]))

                    # step6
                    T.append(X)
                    logger.info('[step 6] exact 1 solution in X:{}'.format(X))
                    logger.info('[step 6] to [step 2]')
                    animation_box.append([
                        (step2_X, _NO_SOLUTIONS_FLAG),
                        (X, _EXACT_1_SOLUTION_FLAG),
                    ])
                    continue  # to step2

        # Tは解が一意に存在するboxのlist
        logger.info('Loop end. cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))
        print('Loop end. cnt:{}, len(S):{}, len(T):{}, len(U):{}'.format(cnt, len(S), len(T), len(U)))
        print
        print cnt
        print('---------- 最終的なS[:10] -----------')
        pprint(S[:10])
        print('---------- 最終的なU[:10] -----------')
        pprint(U[:10])
        print('---------- 最終的なT -----------')
        pprint(T)

        return map(lambda x: self.refine(x), T), S_sizes, T_sizes, U_sizes, animation_box
    

