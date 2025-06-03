#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:54:57 2025

@author: giovannidisarra
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec as grd
import numpy as np
import persim
from itertools import combinations
import copy
from ripser import ripser
import matplotlib as mpl
import time
import pdb
mpl.rcParams['font.family'] = 'Times New Roman'

class Synthetic_Data:

    '''
    A class to generate synthetic data with a definite geometrical structure (circle, sphere or torus)
    
    '''
    
    def __init__(self, topology, dim, num_points):
        
        '''
        Initialize the generator with a specific shape, dimension and number of points. 
        Available combinations are: cirlce with arbitrary dimension, 3D sphere and 3D torus.
        
        Parameters
        ----------
        topology: circle, sphere or torus
        
        dim: dimension of geometrical structure
        
        num_points: number of data points
        
        '''
        
        self.dim = dim
        self.num_points = num_points
        self.topology = topology
        self.coords = np.zeros((self.num_points, self.dim))
        
    
    def generate_dataset(self, a_1, C_1, noise, if_plot):
        
        '''
        Generate synthetic dataset with a given topology, parameters and noise values
        
        Parameters
        ----------
        a_1: circular parameter
        
        C_1: parameter
        
        noise: standard deviation of random gaussian variable added to each data point in n D.
        
        Returns
        ----------
        dim-dimensional dataset
        
        '''
        
        xx = np.zeros(self.num_points)
        yy = np.zeros(self.num_points)
        
        x,y = np.meshgrid( np.linspace(0,1, int(np.sqrt(self.num_points))+1)[:-1],  np.linspace(0,1, int(np.sqrt(self.num_points))+1) )
        x = x.flatten()*2*np.pi
        y = y.flatten()*2*np.pi
        
        if self.topology=='torus':
            if self.dim!=3:
                raise ValueError("Dimension must be 3 to generate data on a torus.")

            for n in range(self.num_points):
                u,v = x[n],y[n]
                u +=np.random.randn()*0.1
                v +=np.random.randn()*0.1
                
                f1 = ( C_1 + a_1 *np.cos( v ) ) * np.cos( u )
                f2 = ( C_1 + a_1 *np.cos( v ) ) * np.sin( u )
                f3 = a_1 * np.sin( v )
                
                self.coords[n,:] = (f1,f2,f3)
                xx[n] = u
                yy[n] = v
                
        if self.topology=='sphere':
            if self.dim!=3:
                raise ValueError("Dimension must be 3 to generate data on a sphere.")
                
            for n in range(self.num_points):
                u,v = x[n],y[n]
                u +=np.random.randn()*0.1
                v +=np.random.randn()*0.1
                
                f1 = C_1 + a_1 * np.sin( v ) * np.cos( u )
                f2 = C_1 + a_1 * np.sin( v ) * np.sin( u )
                f3 = C_1 + a_1 * np.cos( v )
                
                self.coords[n,:] = (f1,f2,f3)
                xx[n] = u
                yy[n] = v
                
        if self.topology=='circle':
            x = np.linspace(0,1, int(self.num_points)+1)[:-1]*2*np.pi
            
            vec1 = np.random.randn(self.dim)
            vec1 = vec1 / np.linalg.norm(vec1)

            vec2 = np.random.randn(self.dim)
            vec2 = vec2 - np.dot(vec2, vec1) * vec1
            vec2 = vec2 / np.linalg.norm(vec2)
            
            for n in range(self.num_points):
                u = x[n]
                
                f = C_1 + a_1 * ( np.cos(u) * vec1 + np.sin(u) * vec2)
                        
                self.coords[n,:] = f
                xx[n] = u
    
        self.coords = self.coords + np.random.normal(0., noise, size = (self.num_points,self.dim))
        
        if if_plot:
            if self.dim==2:
                phi = np.arctan(np.true_divide(self.coords[:,0],self.coords[:,1]))
                phi = self.coords[:,0]
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(111)
                #plt.title('noise='+str(np.round(noise,decimals=2)), fontsize=20)
                #ax.set_axis_off()
                #ax.view_init(60,30)
                c = ax.scatter(self.coords[:,0], self.coords[:,1], s=10,  c=phi, cmap=plt.cm.viridis)
                #plt.savefig('torus_'+str(noise)+'.png', dpi =300, bbox_inches='tight')
                plt.show()
            if self.dim==3:
                phi = np.arctan(np.true_divide(self.coords[:,0],self.coords[:,1]))
                phi = self.coords[:,0]
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(111, projection='3d')
                plt.title('noise='+str(np.round(noise,decimals=2)), fontsize=20)
                #ax.set_axis_off()
                ax.view_init(60,30)
                c = ax.scatter(self.coords[:,0], self.coords[:,1], self.coords[:,2], s=10,  c=phi, cmap=plt.cm.viridis)
                #plt.savefig('torus_'+str(noise)+'.png', dpi =300, bbox_inches='tight')
                plt.show()
                
        return self.coords
        
class Barcode:

    '''
    A class to handle barcodes and compute Gamma
    
    '''
    
    def __init__(self):
            
        '''
        Initialize the barcode
        
        '''
        self.barcode = [np.array([]), np.array([])]
            
    def compute_barcode(self, points, maxdim, if_plot=True):
    
        '''
        Compute barcode from a given dataset 
        
        Parameters
        ----------
        points: input dataset to persistent homology.
        
        maxdim: maximal homological dimension.
        
        '''
        
        out = ripser(points, maxdim=maxdim)
        self.barcode = out['dgms']
        
        if if_plot:
            self.plot(title = 'Barcode')
        
    def plot(self, title = ''):
    
        cs = np.repeat([[0,0.55,0.2]],3).reshape(3,3).T
        alpha = 1
        inf_delta = 0.1
        colormap = cs
        maxdim = len(self.barcode)-1
        dims = np.arange(maxdim+1)

        min_birth, max_death = 0,0            
        for dim in dims:
            persistence_dim = self.barcode[dim][~np.isinf(self.barcode[dim][:,1]),:]
            min_birth = min(min_birth, np.min(persistence_dim))
            max_death = max(max_death, np.max(persistence_dim))
        
        delta = (max_death - min_birth) * inf_delta
        infinity = max_death + delta
        fig = plt.figure(figsize=(6,6))
        gs = grd.GridSpec(len(dims),1)
        
        indsall =  0
        labels = [r"$H_0$", r"$H_1$", r"$H_2$"]
        for dit, dim in enumerate(dims):
            axes = plt.subplot(gs[dim])
            d = np.copy(self.barcode[dim])
            d[np.isinf(d[:,1]),1] = infinity
            dlife = (d[:,1] - d[:,0])
            dinds = np.argsort(dlife)[-30:]
            #if len(dinds)>1:
            #    dl1,dl2 = dlife[dinds[-2:]]
            if dim>0:
                dinds = dinds[np.flip(np.argsort(d[dinds,0]))]
            axes.barh(
                0.5+np.arange(len(dinds)),
                dlife[dinds],
                height=0.8,
                left=d[dinds,0],
                alpha=alpha,
                color=colormap[dim],
                linewidth=0,
            )
            indsall = len(dinds)
            
            plt.suptitle(title, fontsize=20)
            axes.plot([0,0], [0, indsall], c = 'k', linestyle = '-', lw = 1)
            axes.plot([0,indsall],[0,0], c = 'k', linestyle = '-', lw = 1)
            axes.set_xlim([0, infinity])
            #axes.set_xlim([0, 20])
            plt.ylabel(labels[dit], fontsize = 18, rotation = 0, labelpad = 15)
            
            my_xticks = axes.get_xticks()
            plt.xticks([0, my_xticks[-1]])
            plt.yticks(fontsize = 0)
            plt.xticks(fontsize = 18)
            
    def make_idealized(self, betti_n, strict=True, if_plot=True):
    
        '''
        Build idealized reference barcode
        
        Parameters
        ----------
        betti_n: Betti numbers of the reference topology
        
        strict: True or False
    
        Returns
        -------
        final_barcode : idealized reference barcode
        '''
        
        final_barcode = copy.deepcopy(self)
        
        lifetimes = [np.subtract( self.barcode[h][:,1], self.barcode[h][:,0]) for h in range(len(self.barcode))]
        
        for h in range(len(self.barcode)):
            
            if len(self.barcode[h])<2:
                #if betti_n[1] == '1':
                #    final_barcode = copy.deepcopy(self)
                #else:
                self.barcode[h] = np.vstack((self.barcode[h], np.array([0., 0.])))
                final_barcode = copy.deepcopy(self)
            else:
                min_life = np.nanmin(lifetimes[h])
                
                if int(betti_n[h]) == 0:
                    longliveds = sorted(lifetimes[h])[-1:]
                else:
                    longliveds = sorted(lifetimes[h])[-int(betti_n[h]):]
                    
                longlived_min = np.min(longliveds)
                longlived_max = np.max(longliveds)
                
                if int(betti_n[h]) == 0:
                    final_barcode.barcode[h][:,1] = np.where( lifetimes[h] >= longlived_min, self.barcode[h][:,0] + min_life, self.barcode[h][:,1])
                else:
                    
                    if strict:
                        final_barcode.barcode[h][:,1] = np.where( lifetimes[h] >= longlived_min, self.barcode[h][:,0] + longlived_max, self.barcode[h][:,0] + min_life)
                    else:
                        final_barcode.barcode[h][:,1] = np.where( lifetimes[h] >= longlived_min, self.barcode[h][:,1], self.barcode[h][:,0] + min_life)
                
        if if_plot:
            final_barcode.plot(title = 'Idealized barcode for Betti numbers ('+betti_n+')')
            plt.show()
    
        return final_barcode
        
    
    def norm_bdistance(self, barcode2):
    
        '''
        Parameters
        ----------
        barcode2 : second list of holes birth and death times in n-dimensions
    
        Returns
        -------
        nb_dist : normalized bottleneck distance between the two lists in n-dimensions.
        '''
        
        barcodes = [copy.deepcopy(self.barcode), copy.deepcopy(barcode2.barcode)]
        
        dims = len(barcodes[0])
        diams = np.zeros((len(barcodes), dims))
        
        #compute diameters of the barcodes in n dimensions (dims)
        #for i,barcode in enumerate(barcodes):
        #    for j in range(1,dims):
                 
        #        diams[i][j] = np.max(np.array([ persim.bottleneck([bd[0]],[bd[1]]) for bd in list(combinations(barcode[j],2))]))


        for i,barcode in enumerate(barcodes):
            for j in range(1,dims):
                bar_max = barcode[j][np.where(np.diff(barcode[j], axis=1)==np.min(np.diff(barcode[j], axis=1)))[0][0]]
                bar_min = barcode[j][np.where(np.diff(barcode[j], axis=1)==np.max(np.diff(barcode[j], axis=1)))[0][0]]
     
                diams[i][j] =  persim.bottleneck([bar_max], [bar_min])
        
        for i,barcode in enumerate(barcodes):
            for j,bars in enumerate(barcode):
                if j==0:
                    continue
                else:
                    barcodes[i][j] = bars / diams[i][j]
        
        nb_dist = np.array([ persim.bottleneck(barcodes[0][j], barcodes[1][j] ) for j in range(1,dims)])[0]
        
        return nb_dist
    
    
    def compute_Gamma(self, ref, betti_n = '121'):
    
        '''
        Parameters
        ----------
        ref: reference to compute Gamma (arbitrary barcode or 'self')
        
        betti_n: betti numbers of the reference topology
    
        Returns
        -------
        Gamma : degree of topology in n dimensions for barcode
        '''
        
        ideal_barcode = self.make_idealized( betti_n, strict=True, if_plot=False)
        
        if ref == 'self':
            Gamma = 1 - self.norm_bdistance( ideal_barcode)
        else:
            Gamma = 1 - self.norm_bdistance( ref)
        
        return Gamma
