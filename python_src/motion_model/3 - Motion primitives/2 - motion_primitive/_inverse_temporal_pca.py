    def _inverse_temporal_pca(self, gamma):
        """ Backtransform a lowdimensional vector gamma to the timewarping
        function t'(t)

        Parameters
        ----------
        * gamma: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * time_function: numpy.ndarray
        \tThe indices of the timewarping function t'(t)
        """
        # reconstruct harmonics and meanfd from coefs
        fd = robjects.r['fd']

        basis = self.t_pca["basis_function"]

        eigen_coefs = numpy2ri.numpy2ri(self.t_pca["eigen_vectors"])
        eigenfd = fd(eigen_coefs, basis)
        mean_coefs = numpy2ri.numpy2ri(self.t_pca["mean_vector"])
        meanfd = fd(mean_coefs, basis)
        numframes = self.n_canonical_frames

        # reconstruct t(t') from gamma
        fdeval = robjects.r['eval.fd']

        t = []
        t.append(0)
        for i in xrange(numframes):
            mean_i = fdeval(i, meanfd)
            mean_i = np.ravel(np.asarray(mean_i))[-1]
            eigen_i = np.asarray(fdeval(i, eigenfd))[0]    # its a nested array
            t.append(t[-1] + np.exp(mean_i + np.dot(eigen_i, gamma)))

        # undo step from timeVarinaces.transform_timefunction during alignment
        t = np.array(t[1:])
        t -= 1
        zeroindices = t < 0
        t[zeroindices] = 0

        # calculate inverse spline by creating a spline, upsampling it and
        # use the samples to get an inverse spline
        # i.e. calculate t'(t) from t(t')
        T = len(t) - 1
        x = np.linspace(0, T, T+1)
        spline = UnivariateSpline(x, t, s=0, k=2)

        x_sample = np.linspace(0, T, 200)
        w_sample = spline(x_sample)

        # try to get a valid inverse spline
        s = 10
        frames = np.linspace(1, t[-1], np.round(t[-1])-1)
        while True:
            inverse_spline = UnivariateSpline(w_sample, x_sample, s=s, k=2)
            if not np.isnan(inverse_spline(frames)).any():
                break
            s = s + 1

        frames = np.linspace(1, t[-1], np.round(t[-1]))
        t = inverse_spline(frames)
        t = np.insert(t, 0, 0)
        t = np.insert(t[:-1], len(t)-1, numframes-1)
        return t