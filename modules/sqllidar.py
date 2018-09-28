#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sqlite3
import csv
import os
import itertools

# https://stackoverflow.com/questions/26464567/csv-read-specific-row
# https://w3.leica-geosystems.com/kb/?guid=5532D590-114C-43CD-A55F-FE79E5937CB2

class Lidarsql():
    def __init__(self, path='..\db\lidar.sql'):
        self.path = path
        self.connect()
        self.create_db()

    def connect(self):
        self.conn = sqlite3.connect(self.path)
        self.curs = self.conn.cursor()

    def close(self):
        self.conn.close()

    def create_db(self):
        sceneinstruction = """CREATE TABLE IF NOT EXISTS scene (
                                id integer PRIMARY KEY,
                                comment text NULL,
                                columns integer,
                                rows integer
                                );
                            """
        self.curs.execute(sceneinstruction)

        cloudinstrcution = """CREATE TABLE IF NOT EXISTS cloud (
                            id integer PRIMARY KEY,
                            scene integer NOT NULL,
                            X REAL NOT NULL,
                            Y REAL NOT NULL,
                            Z REAL NOT NULL,
                            Intensity  REAL NOT NULL,
                            dist REAL,
                            theta REAL,
                            phi REAL,
                            R INTEGER,
                            G INTEGER,
                            B INTEGER,
                            FOREIGN KEY(scene) REFERENCES scene(id)
                            );"""
        self.curs.execute(cloudinstrcution)

    def select(self):
        pass

    def write_cloud(self, cloud, scene=None):
        if scene is None:
            self.curs.execute('SELECT id FROM scene ORDER BY id DESC LIMIT 1;')
            scene = self.curs.fetchone()[0]

        sphericals = self.get_spherical(np.array(cloud)[:,:3])
        coords = np.hstack((cloud, sphericals))
        self.curs.executemany('''INSERT INTO cloud (scene, X, Y, Z, Intensity, R, G, B, dist, theta, phi) VALUES ({},?,?,?,?,?,?,?,?,?,?)'''.format(scene), coords.tolist())
        print('done')


    def write_scene(self, scene):
        scene_instruction = """
            INSERT INTO scene (comment, columns, rows)
            VALUES ('{}', {}, {});
            """.format(scene['comment'], scene['columns'], scene['rows'])
        self.curs.execute(scene_instruction)

    def create_scene(self, scene, cloud):
        self.write_scene(scene)
        self.write_cloud(cloud)
        self.conn.commit()

    def get_spherical(self, xyz):
        ptsnew = np.zeros((xyz.shape[0], 3))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # theta
        ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])  # phi
        return ptsnew



class Ptxreader():
    def __init__(self, filepath):
        self.scenes = {} #the starting line for each scene
        self.filepath = filepath #PTX file location

    def get_digits(self, entry):
        """ Converts a list of stings into a list of floats"""
        return [float(x) for x in entry]

    def get_scene(self, scenenumber):
        """Extract all the scene information"""
        scene_params = self.get_scene_params(scenenumber)
        cloud = self.get_cloud(scenenumber)

        return scene_params, cloud

    def get_scene_params(self, scenenumber):
        """Returns the scene parameters"""
        scene_params = {}

        firstline = self.scenes[scenenumber]
        lastline = firstline + 10

        with open(self.filepath, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            header = []
            for row in itertools.islice(reader, firstline, lastline):
                header.append(self.get_digits(row))

        scene_params['columns'] = int(header[0][0])
        scene_params['rows'] = int(header[1][0])
        scene_params['position'] = header[2]
        scene_params['regposition'] = header[3:6]
        scene_params['transformation'] = header[6:]
        return scene_params

    def get_cloud(self, scenenumber):
        """ Read the lines for a point cloudinstrcution """
        firstline = self.scenes[scenenumber]+11
        if scenenumber+1 in self.scenes.keys():
            lastline = self.scenes[scenenumber+1]
        else:
            lastline = None

        cloud = []
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file, delimiter=' ')

            cloudgen = itertools.filterfalse(lambda x: x ==  ['0', '0', '0', '0.500000', '0', '0', '0'],
                                        itertools.islice(reader, firstline, lastline))
            cloud = [self.get_digits(item) for item in cloudgen]

        #print(len(cloud))
        return cloud

    def get_scene_list(self):
        """ Reads the PTX file to list all the scenes available """
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            scenestarts = []
            for index, line in enumerate(reader):
                if len(line) == 1 and index-1 not in scenestarts:
                    scenestarts.append(index)
                    self.scenes[len(self.scenes)] = index
                    print('scene {} detected'.format(len(self.scenes)))

                if index % 1000000 == 0:
                    print('{:5d} processed'.format(index))

        print(self.scenes)


def main1():
    lidarsql = Lidarsql()

    sample = [[-1.278275, -0.97258, -1.599747, 0.255985, 160.0, 160.0, 106.0],
              [-1.281601, -0.975113, -1.600754, 0.267948, 155.0, 158.0, 105.0],
              [-1.28331, -0.976425, -1.599625, 0.288945, 134.0, 136.0, 87.0],
              [-1.286087, -0.978531, -1.599899, 0.264042, 165.0, 164.0, 110.0],
              [-1.290421, -0.981827, -1.602066, 0.282109, 164.0, 162.0, 103.0],
              [-1.292801, -0.983627, -1.601822, 0.241093, 182.0, 182.0, 112.0],
              [-1.295029, -0.985336, -1.601395, 0.250126, 143.0, 142.0, 88.0],
              [-1.297012, -0.986832, -1.600601, 0.249149, 124.0, 123.0, 77.0],
              [-1.302444, -0.990982, -1.604111, 0.267948, 119.0, 120.0, 76.0],
              [-1.304276, -0.992386, -1.603134, 0.23035, 135.0, 136.0, 94.0]]

    #lidarsql.write_scene(scene={'rows': 50, 'columns':150, 'comment':'Bryce1'})
    lidarsql.create_scene(scene={'rows': 50, 'columns':150, 'comment':'Bryce1'}, cloud=sample)
    lidarsql.close()

def main2():
    ptxreader = Ptxreader('../data/Bryce.ptx')
    #ptxreader.get_scene_list()
    ptxreader.scenes = {0: 0, 1: 9862750, 2: 19725500, 3: 29575686, 'last': 39438435}
    scene_params, cloud = ptxreader.get_scene(1)
    print(scene_params)
    print('done')

def main3():
    ptxreader = Ptxreader('../data/Bryce.ptx')
    ptxreader.scenes = {0: 0, 1: 9862750, 2: 19725500, 3: 29575686}
    lidarsql = Lidarsql()

    for sceneindex in ptxreader.scenes.keys():
        print(sceneindex)
        scene_params, cloud = ptxreader.get_scene(sceneindex)
        lidarsql.create_scene({**scene_params, 'comment':'Bryce - {}'.format(sceneindex)}, cloud=cloud)

if __name__ == '__main__':
    main3()
