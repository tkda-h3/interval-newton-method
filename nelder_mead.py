# -*- coding: utf-8 -*-
#!/usr/bin/env python

import datetime
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from  matplotlib.tri import Triangulation
import os

class NelderMead():
    """
        minimize func
    """
    def __init__(self, func, points, alpha=1, gamma=2, rho=0.5, sigma=0.5):
        self.func = func 
        self.points = points
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.points_history = None

    @property
    def x_o(self): # centroid(重心)
        return np.mean(self.points[:-1], axis=0)
    
    @property
    def x_r(self):
        x_r = self.x_o + self.alpha * (self.x_o - self.points[-1])
        return x_r  
    
    @property
    def x_e(self):
        x_e = self.x_o + self.gamma * (self.x_r - self.x_o)
        return x_e
    
    @property
    def x_c(self):
        x_c = self.x_o + self.rho * (self.x_o - self.points[-1])
        return x_c
    
    @property 
    def points(self):
        return self.points
    
    def reflection(self):
        self.points[-1] = self.x_r
        
    def expansion(self):
        if self.func(self.x_e) < self.func(self.x_r):
            self.points[-1] = self.x_e
        else:
            self.points[-1] = self.x_r
    
    def contraction_and_shrink(self):
        if self.func(self.x_c) < self.func(self.points[-1]):
            self.points[-1] = self.x_c
        else:
            self.points = self.points[0] + self.sigma * (self.points - self.points[0])
            
    def sort_points(self):
        self.points = np.array(sorted(self.points, key=lambda x: self.func(x)))
        return self.points
    
    def update_points(self): # 1回更新する
        self.sort_points()
        if self.func(self.points[0]) <= self.func(self.x_r) < self.func(self.points[-1]):
            self.reflection()
        elif self.func(self.x_r) < self.func(self.points[0]):
            self.expansion()
        else:
            self.contraction_and_shrink()
        return self.points

    def run(self, times=30):
        """
        times: itertion times
        """
        self.points_history = [self.points]
        for i in range(1, 50+1):
            self.update_points()
            self.points_history.append(self.points)
        self.sort_points()
        minima = self.func(self.points[0])
        return minima, self.points_history
    
    def save_fig(self, X, Y, Z, points, filepath, title):
        fig = plt.figure(figsize=(10,8))
        im = plt.contour(X, Y, Z, 20, alpha=0.55, zorder=-1, shading='gouraud')
        plt.colorbar(im)
        triang=  Triangulation(*points.T)
        plt.triplot(triang, 'bo-')
        plt.xlim((np.min(X),np.max(X)))
        plt.ylim((np.min(Y),np.max(Y)))
        plt.title(title)
        fig.savefig(filepath)
        plt.close(fig)

    def save_figs(self, X, Y, Z, points_list):
        now_string = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        dirpath = 'nelder_mead_image/{}'.format(now_string)
        os.makedirs(dirpath)
        print("Start saveing images in {} dirctory.".format(dirpath))
        for i, points in enumerate(points_list):
            image_path = os.path.join(dirpath, "anime_{0:0>4}.png".format(i))
            title = 'cnt:{}'.format(i)
            self.save_fig(X, Y, Z, points, image_path, title)
        print("Finish saveing images in {} dirctory.".format(dirpath))
    
            
if __name__ == '__main__':
    from ivmat import ivmat as ip
    from interval import (
        interval,
        imath,
    )
    from sympy import(
        exp,
        sin,
        cos,
        var,
        lambdify,
    )
    import datetime
    import os
    
    seed_num = 2222
    np.random.seed(seed_num)
    low = -2.5
    high = 2.5
    points = np.random.uniform(low=low, high=high, size=(3,2))
    print points
    
    x_1, x_2 = args = var('x_1, x_2')
    f_expr = 0.6*exp(-4*(x_1-2*x_2)**2 - 6*(x_2-0.5)**2) + exp(-7*(cos(2*x_1) - (x_1 - sin(x_2)))**2 - 9*(0.9*cos(x_2))**2)
    f = lambdify([args], -f_expr, modules=np)        
    
    x = np.arange(-2, 2, 0.05) 
    y = np.arange(-2, 2, 0.05) 
    X, Y = np.meshgrid(x, y)
    args_list =  np.array([X.flatten(), Y.flatten()]).T[:, :, np.newaxis].tolist()
    Z = np.array([-f(_args) for _args in args_list]).reshape(X.shape)
    
    nelder = NelderMead(f, points)
    now_string = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    dirpath = 'nelder_mead_image/{}'.format(now_string)
    os.makedirs(dirpath)
    
    for i in range(1, 30+1):
        nelder.update_points()
        image_path = os.path.join(dirpath, "anime_{0:0>4}.png".format(i))
        title = 'cnt:{}'.format(i)
        save_fig(X,Y,Z, nelder.points, image_path, title)
