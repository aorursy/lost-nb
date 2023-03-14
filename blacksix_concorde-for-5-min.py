#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy




get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./linkern ]]; then\n  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz\n  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c\n  tar xf co031219.tgz\n  (cd concorde && CFLAGS='-Ofast -march=native -mtune=native -fPIC' ./configure)\n  (cd concorde/LINKERN && make -j && cp linkern ../../)\n  rm -rf concorde co031219.tgz\nfi")




def read_cities(filename='../input/cities.csv'):
    return pd.read_csv(filename, index_col=['CityId'])

cities = read_cities()




cities1k = cities * 1000




def write_tsp(cities, filename, name='traveling-santa-2018-prime-paths'):
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index+1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities1k, 'cities1k.tsp')




get_ipython().run_cell_magic('bash', '', 'time ./linkern -K 1 -s 42 -S linkern.tour -R 999999999 -t 300 ./cities1k.tsp >linkern.log')




get_ipython().system("sed -Ene 's/([0-9]+) Steps.*Best: ([0-9]+).*/\\1,\\2/p' linkern.log >linkern.csv")
pd.read_csv('linkern.csv', index_col=0, names=['TSP tour length']).plot();




class Tour:
    cities = read_cities()
    coords = (cities.X + 1j * cities.Y).values
    penalized = ~cities.index.isin(sympy.primerange(0, len(cities)))

    def __init__(self, data):
        """Initializes from a list/iterable of indexes or a filename of tour in csv/tsplib/linkern format."""

        if type(data) is str:
            data = self._read(data)
        elif type(data) is not np.ndarray or data.dtype != np.int32:
            data = np.array(data, dtype=np.int32)
        self.data = data

        if (self.data[0] != 0 or self.data[-1] != 0 or len(self.data) != len(self.cities) + 1):
            raise Exception('Invalid tour')

    @classmethod
    def _read(cls, filename):
        data = open(filename, 'r').read()
        if data.startswith('Path'):  # csv
            return pd.read_csv(io.StringIO(data)).Path.values
        offs = data.find('TOUR_SECTION\n')
        if offs != -1:  # TSPLIB/LKH
            data = np.fromstring(data[offs+13:], sep='\n', dtype=np.int32)
            data[-1] = 1
            return data - 1
        else:  # linkern
            data = data.replace('\n', ' ')
            data = np.fromstring(data, sep=' ', dtype=np.int32)
            if len(data) != data[0] + 1:
                raise Exception('Unrecognized format in %s' % filename)
            return np.concatenate((data[1:], [0]))

    def info(self):
        dist = np.abs(np.diff(self.coords[self.data]))
        penalty = 0.1 * np.sum(dist[9::10] * self.penalized[self.data[9:-1:10]])
        dist = np.sum(dist)
        return { 'score': dist + penalty, 'dist': dist, 'penalty': penalty }

    def dist(self):
        return self.info()['dist']

    def score(self):
        return self.info()['score']

    def __repr__(self):
        return 'Tour: %s' % str(self.info())

    def to_csv(self, filename):
        pd.DataFrame({'Path': self.data}).to_csv(filename, index=False)




tour = Tour('linkern.tour')
tour




tour.to_csv('submission.csv')




def plot_tour(tour, cmap=mpl.cm.gist_rainbow, figsize=(25, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(tour.data)

    for i in range(201):
        ind = tour.data[n//200*i:min(n, n//200*(i+1)+1)]
        ax.plot(tour.cities.X[ind], tour.cities.Y[ind], color=cmap(i/200.0), linewidth=1)

    ax.plot(tour.cities.X[0], tour.cities.Y[0], marker='*', markersize=15, markerfacecolor='k')
    ax.autoscale(tight=True)
    mpl.colorbar.ColorbarBase(ax=fig.add_axes([0.125, 0.075, 0.775, 0.01]),
                              norm=mpl.colors.Normalize(vmin=0, vmax=n),
                              cmap=cmap, orientation='horizontal')

plot_tour(tour)

