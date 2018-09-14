#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class Elevationcorr():
    """
    Corrects the elevation of a LIDAR sample by using the result of a linear regression
    (can be limited to a certain area). The

    dataset --> Pandas dataframe containing the LIDAR sample
    coordx  --> colname used as X (entry) for the linear regression
    coordy  --> colname used as y (target) for the linear regression
    elevation --> elevation parameter to Correct
    xrange --> Used to limit the area where the regression will be fitted,
                if None, no limitation in the X dimension
                if (xmin, xmax), dataset cropped
    """

    def __init__(self, coordx='x', coordy='y', elevation='z', xrange=None, yrange=None, suffix_corr='_corr'):
        self.coordx = coordx
        self.coordy = coordy
        self.elevation = elevation
        self.xrange = xrange
        self.yrange = yrange
        self.trained = False
        self.slope = 1e20
        self.offset = 1e20
        self.suffix_corr = suffix_corr

    def fit(self, dataset):
        # dataset creation used for the linear regression
        self.dataset = dataset
        df = self.dataset.copy()
        if self.xrange is not None:
            df = df[df[self.coordx] > self.xrange[0]]
            df = df[df[self.coordx] < self.xrange[1]]
        if self.yrange is not None:
            df = df[df[self.coordy] > self.yrange[0]]
            df = df[df[self.coordy] < self.yrange[1]]

        # Offset and slope calculation
        self.offset = (df[self.elevation].max() - df[self.elevation].min())
        self.slope = np.degrees(np.arctan(self.offset/np.abs((dataset[self.coordx].max() - dataset[self.coordx].min()))))

        # linear regression fitting
        self.slopereg = LinearRegression()
        self.slopereg.fit(X=dataset[self.coordx].values.reshape(-1,1), y=dataset[self.elevation].values.reshape(-1,1))

        self.trained = True
        return self

    def transform(self, dataset):
        if self.trained:
            dataset['z{}'.format(self.suffix_corr)] = dataset[self.elevation] - self.slopereg.predict(dataset[self.coordx].values.reshape(-1,1)).reshape(-1)
            return dataset
        return "Model not trained"

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)


    def get_slope(self):
        if self.trained:
            return self.slope
        return 'Model not trained'

    def get_offset(self):
        if self.trained:
            return self.offset
        return 'Model not trained'

if __name__ == '__main__':
    colsname = ['x', 'y', 'z', 'xn', 'yn', 'zn', 'R', 'G', 'B', 'A', 'quality', 'zz']
    lidar = pd.read_csv('../data/20180911_SHCCurb3.ply',
                    delimiter=' ',
                    skiprows=18,
                    #nrows=500000, #use None to read the complete dataset or the number of first rows to read
                    header=None,
                    )
    lidar.columns=colsname
    lidar.drop(['zz'], inplace=True, axis=1)

    elevationcorr = Elevationcorr()
    print(elevationcorr.fit_transform(lidar).shape)
    print(elevationcorr.get_slope())
    print(elevationcorr.get_offset())
