from ivmat import ivmat as ip


class fmat(list):
    def __call__(self, X_mat):
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
        return ip(ans)
