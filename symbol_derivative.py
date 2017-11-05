# -*- coding: utf-8 -*-
#!/usr/bin/env python

import datetime
import os

from interval import (
    imath,
)
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from sympy import (
    diff,
)
from sympy.utilities.lambdify import lambdify

from consts import COLOR_MAP
from fmat import fmat
from ivmat import ivmat as ip
from krawczyk import Krawczyk


def get_f_df_ddf_from_symbol_representation(f_expr, args):
    """
    Params:
        f_expr: 関数fのsympyのsymbol representation
        args: fの引数のtuple
    
    Return:
        f: fのlambdaを要素とするfmat object
        df: dfのlambdaを要素とするfmat object
        ddf: ddfのlambdaを要素とするfmat object
    """
    f_func = lambdify(args, f_expr, modules=imath)
    f = fmat([[f_func]])

    df_expr = [diff(f_expr, x) for x in args]
    df_list = [[lambdify(args, _df_expr, modules=imath)] for _df_expr in df_expr]
    df = fmat(df_list)

    ddf_expr = [[diff(_df_expr, x) for x in args] for _df_expr in df_expr]
    ddf_list = [[lambdify(args, _ddf_expr, modules=imath) for _ddf_expr in _row] for _row in ddf_expr]
    ddf = fmat(ddf_list)
    return f, df, ddf


def calc_f_meshgrid(f, X_init):
    """
    背景のcontour mapのデータ準備
    Return:
        X, Y = np.meshgrid(x,y)
        Z: X.shapeに対応した関数値
        x_lim: x軸の探索区間
        y_lim: y軸の探索区間
    """
    x = np.arange(X_init[0][0][0].inf, X_init[0][0][0].sup, 0.05) 
    y = np.arange(X_init[1][0][0].inf, X_init[1][0][0].sup, 0.05) 
    X, Y = np.meshgrid(x, y)
    args_list =  np.array([X.flatten(), Y.flatten()]).T[:, :, np.newaxis].tolist()
    Z = np.array([ip.mid(f(_args))[0][0] for _args in args_list]).reshape(X.shape)

    x_lim = (X_init[0][0][0].inf, X_init[0][0][0].sup)
    y_lim = (X_init[1][0][0].inf, X_init[1][0][0].sup)
    return X, Y, Z, x_lim, y_lim


def visualize_optimization_log(krawczyk_obj, f, animation_box, skip=5, title_prefix=''):
    """
    krawczyk_obj: Krawczyk class object
    f: 最適化する関数f
    animation_box: 探索を可視化するためのログ。krawczyk_obj.find_global_minimaの戻り値
    skip: animationのcountのスキップ幅
    title_prefix: matplotlib.figureのタイトルのprefix
    """
    def get_rect(x_1, x_2, facecolor_code, edgecolor_code):
        left, right, below, above = x_1[0][0], x_1[0][1], x_2[0][0], x_2[0][1]
        rect = Rectangle((left, below),
                        right - left,
                        above - below,
                        facecolor= facecolor_code,
                        edgecolor=edgecolor_code
                        )
        return rect

    
    X, Y, Z, x1_lim, x2_lim = calc_f_meshgrid(f, krawczyk_obj.X_init)
    now_string = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    dirpath = 'image/{}'.format(now_string)
    os.makedirs(dirpath)
    print("Start saveing images in {} dirctory.".format(dirpath))

    for i in range(len(animation_box)):
        if i != len(animation_box) - 1: #最後以外
            if i % skip:
                continue
        fig = plt.figure(figsize=(8,6))
        plt.title(title_prefix + ' count: {}'.format(i))
        plt.xlim(x1_lim)
        plt.ylim(x2_lim)
        ax = fig.add_subplot(111)
        for j in range(i+1):
            for parent_box, parent_flag in animation_box[j]:
                fcolor, ecolor = COLOR_MAP[parent_flag]# facecolor, edgecolor
                parent_x1 = parent_box[0][0]
                parent_x2 = parent_box[1][0]
                if ip.is_empty(parent_box):
                    continue
                rect = get_rect(parent_x1, parent_x2, fcolor, ecolor)
                ax.add_patch(rect)

        # 最後にcontour mapを薄く重ねる    
        im = plt.contour(X, Y, Z, alpha=0.8, zorder=100, shading='gouraud')
        fig.colorbar(im)
        image_path = os.path.join(dirpath, "anime_{0:0>4}.png".format(i))
        fig.savefig(image_path)
        plt.close(fig)
    print("Finish saveing images in {} dirctory.".format(dirpath))


def get_global_minima_from_f_expr(f_expr, args, X):
    """
    f_expr: sympy representation
    args: iterable object of sympy symbol
    X: ivmat object. This box is an initial exploration range.
    
    実行時のパラメータ調整が必要な場合は自分で書いて下さい。
    """
    f, df, ddf = get_f_df_ddf_from_symbol_representation(f_expr, args)
    krawczyk = Krawczyk(df, ddf, X)
    cnt_max = 2000
    max_width = 1e-4
    ans_boxes, S_num_list, T_num_list, U_num_list, animation_box = krawczyk.find_global_minimum(f, trace=False, cnt_max=cnt_max, max_width=max_width)
    visualize_optimization_log(krawczyk, f, animation_box)    

    
if __name__ == '__main__':
    from interval import interval
    from sympy import(
        exp,
        var,        
    )
        
    x_1, x_2 = args = var("x_1 x_2")
    f_expr = -(exp(-4*(x_1-1)**2 - 6*(x_2-0.5)**2) + exp(-7*(x_1+1.5)**2 - 9*(x_2+1)**2))
    f, df, ddf = get_f_df_ddf_from_symbol_representation(f_expr, args)
    
    X = ip([[interval[-2, 2.2]],[interval[-2, 2.2]]])
    krawczyk = Krawczyk(df, ddf, X)
    cnt_max = 2000
    max_width = 1e-4
    ans_boxes, S_num_list, T_num_list, U_num_list, animation_box = krawczyk.find_global_minimum(f, trace=False, cnt_max=cnt_max, max_width=max_width)
    visualize_optimization_log(krawczyk, f, animation_box)
