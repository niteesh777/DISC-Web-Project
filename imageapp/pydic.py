#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of pydic, a free digital correlation suite for computing strain fields
#
# Author :  - Damien ANDRE, SPCTS/ENSIL-ENSCI, Limoges France
#             <damien.andre@unilim.fr>
#
# Copyright (C) 2017 Damien ANDRE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



# ====== INTRODUCTION
# Welcome to pydic a free python suite for digital image correlation.
# pydic allows to compute (smoothed or not) strain fields from a serie of pictures.
# pydic takes in account the rigid body transformation.

# Note that this file is a module file, you can't execute it.
# You can go to the example directory for usage examples.


import numpy as np
import cv2
import sys
import math
import glob
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cv2
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import scipy.interpolate
import copy
import os
import pandas as pd



grid_list = [] # saving grid here

class grid:
     """The grid class is the main class of pydic. This class embed a lot of usefull
method to treat and post-treat results"""

     def __init__(self, grid_x, grid_y, size_x, size_y):
          """Construct a new grid objet with x coordinate (grid_x),
y coordinate (grid_y), number of point along x (size_x) and
number of point along y (size_y)"""
          self.grid_x = grid_x
          self.grid_y = grid_y
          self.size_x = size_x
          self.size_y = size_y
          self.disp_x =  self.grid_x.copy().fill(0.)
          self.disp_y =  self.grid_y.copy().fill(0.)
          #try to add disp_xy
          self.disp_xy_indi = self.grid_x.copy().fill(0.)
          self.strain_xx = None
          self.strain_yy = None
          self.strain_xy = None

     def add_raw_data(self, winsize, reference_image, image, reference_point, correlated_point, disp):
          """Save raw data to the current grid object. These raw data are used as initial data
for digital image correlation"""

          self.winsize = winsize
          self.reference_image = reference_image
          self.image = image
          self.reference_point = reference_point
          self.correlated_point = correlated_point
          self.disp = disp

     def add_meta_info(self, meta_info):
          """Save the related meta info into the current grid object"""
          self.meta_info = meta_info

     def prepare_saved_file(self, prefix, extension):
          """Not documented, for internal use only"""

          folder = os.path.dirname(self.image)
          folder = folder + '/pydic/' + prefix
          if not os.path.exists(folder):os.makedirs(folder)
          base = os.path.basename(self.image)
          name = folder + '/' + os.path.splitext(base)[0] + '_' + prefix + '.' + extension
          print("saving", name, "file...")
          return name

     def draw_marker_img(self):
          """Draw marker image"""
          name = self.prepare_saved_file('marker', 'png')
          draw_opencv(self.image, point=self.correlated_point, l_color=(0,0,255), p_color=(255,255,0), filename=name, text=name)

     def draw_disp_img(self, scale):
          """Draw displacement image. A scale value can be passed to amplify the displacement field"""
          name = self.prepare_saved_file('disp', 'png')
          ##draw_opencv(self.reference_image, point=self.reference_point, pointf=self.correlated_point, l_color=(0,0,255), p_color=(255,255,0), scale=scale, filename=name, text=name)
          draw_opencv(self.image, point=self.reference_point, pointf=self.correlated_point, l_color=(0,0,255), p_color=(0,0,255), scale=scale, filename=name, text='Amplification = '+str(scale))




     def draw_disp_hsv_img(self, *args, **kwargs):
          """Draw displacement image in a hsv view."""
          name = self.prepare_saved_file('disp_hsv', 'png')
          img = self.reference_image
          if type(img) == str :
               img = cv2.imread(img, 0)


          disp = self.correlated_point - self.reference_point
          fx, fy = disp[:,0], disp[:,1]
          v_all = np.sqrt(fx*fx + fy*fy)
          v_max = np.mean(v_all) + 2.*np.std(v_all)


          rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
          hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

          if v_max != 0.:
               for i, val in enumerate(self.reference_point):
                    disp = self.correlated_point[i] - val
                    ang = np.arctan2(disp[1], disp[0]) + np.pi
                    v = np.sqrt(disp[0]**2 + disp[1]**2)
                    pt_x = int(val[0])
                    pt_y = int(val[1])

                    hsv[pt_y,pt_x, 0] = int(ang*(180/np.pi/2))
                    hsv[pt_y,pt_x, 1] = 255 if int((v/v_max)*255.) > 255 else int((v/v_max)*255.)
                    hsv[pt_y,pt_x, 2] = 255 if int((v/v_max)*255.) > 255 else int((v/v_max)*255.)


          bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
          bgr = cv2.putText(bgr, name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)

          if 'save_img' in kwargs:
               cv2.imwrite(name, bgr)
          if 'show_img' in kwargs:
               cv2.namedWindow('image', cv2.WINDOW_NORMAL)
               cv2.resizeWindow('image', bgr.shape[1], bgr.shape[0])
               cv2.imshow('image', bgr)
               cv2.waitKey(0)
               cv2.destroyAllWindows()




     def draw_grid_img(self, scale):
          """Draw grid image. A scale value can be passed to amplify the displacement field"""
          name = self.prepare_saved_file('grid', 'png')
          draw_opencv(self.reference_image, grid = self, scale=scale, gr_color=(255,255,250), filename=name, text=name)

     def write_result(self):
          """write a raw csv result file. Indeed, you can use your favorite tool to post-treat this file"""
          name = self.prepare_saved_file('result', 'csv')
          f = open(name, 'w')
          f.write("pos_x"    + ',' + "pos_y"    + ',' +
                  "disp_x"    + ',' + "disp_y"  + ',' + "disp_xy_indi" + '\n')
          index = 0
          for i in range(self.size_x):
            for j in range(self.size_y):
                 f.write(str(self.grid_x[i,j])    + ',' + str(self.grid_y[i,j])    + ',' +
                         str(self.disp_x[i,j])    + ',' + str(self.disp_y[i,j])    + ',' + str(self.disp_xy_indi[i,j]) + '\n')
                 index = index + 1
          f.close()


     def plot_field(self, field, title):
          """Plot the chosen field such as strain_xx, disp_xx, etc. in a matplotlib interactive map"""
          image_ref = cv2.imread(self.image)
          ##image_ref = cv2.imread(self.image, 0)
          new_img_ref = cv2.cvtColor(image_ref, cv2.COLOR_RGB2BGR)
          Plot(new_img_ref, self, field, title)

     def interpolate_displacement(self, point, disp, *args, **kwargs):
          """Interpolate the displacement field. It allows to (i) construct the displacement grid and to
(ii) smooth the displacement field thanks to the chosen method (raw, linear, spline,etc.)"""

          x = np.array([p[0] for p in point])
          y = np.array([p[1] for p in point])
          dx = np.array([d[0] for d in disp])
          dy = np.array([d[1] for d in disp])
          method = 'linear' if not 'method' in kwargs else kwargs['method']

          print('interpolate displacement with', method, 'method')
          if method=='delaunay':
               from scipy.interpolate import LinearNDInterpolator
               inter_x = LinearNDInterpolator(point, dx)
               inter_y = LinearNDInterpolator(point, dy)
               self.disp_x = inter_x(self.grid_x, self.grid_y)
               self.disp_y = inter_y(self.grid_x, self.grid_y)

          elif method=='raw':
               # need debugging
               self.disp_x = self.grid_x.copy()
               self.disp_y = self.grid_y.copy()
               self.disp_xy_indi = self.grid_x.copy()

               assert self.disp_x.shape[0] == self.disp_y.shape[0], "bad shape"
               assert self.disp_x.shape[1] == self.disp_y.shape[1], "bad shape"
               assert len(dx) == len(dy), "bad shape"
               assert self.disp_x.shape[1]*self.disp_x.shape[0] == len(dx), "bad shape"
               count = 0
               for i in range(self.disp_x.shape[0]):
                    for j in range(self.disp_x.shape[1]):
                         self.disp_x[i,j] = dx[count]
                         self.disp_y[i,j] = dy[count]
                         self.disp_xy_indi[i,j] = (dx[count]**2+dy[count]**2)**0.5
                         count = count + 1

          elif method=='spline':
               tck_x = scipy.interpolate.bisplrep(self.grid_x, self.grid_y, dx, kx=5, ky=5)
               self.disp_x = scipy.interpolate.bisplev(self.grid_x[:,0], self.grid_y[0,:],tck_x)

               tck_y = scipy.interpolate.bisplrep(self.grid_x, self.grid_y, dy, kx=5, ky=5)
               self.disp_y = scipy.interpolate.bisplev(self.grid_x[:,0], self.grid_y[0,:],tck_y)

          else:
               self.disp_x = griddata((x, y), dx, (self.grid_x, self.grid_y), method=method)
               self.disp_y = griddata((x, y), dy, (self.grid_x, self.grid_y), method=method)



     def compute_strain_field(self):
          """Compute strain field from displacement thanks to numpy"""
          #get strain fields
          dx = self.grid_x[1][0] - self.grid_x[0][0]
          dy = self.grid_y[0][1] - self.grid_y[0][0]


          strain_xx, strain_xy = np.gradient(self.disp_x, dx, dy, edge_order=2)
          strain_yx, strain_yy = np.gradient(self.disp_y, dx, dy, edge_order=2)

          self.strain_xx = strain_xx + .5*(np.power(strain_xx,2) + np.power(strain_yy,2))
          self.strain_yy = strain_yy + .5*(np.power(strain_xx,2) + np.power(strain_yy,2))
          self.strain_xy = .5*(strain_xy + strain_yx + strain_xx*strain_xy + strain_yx*strain_yy)


     def compute_strain_field_DA(self):
          """Compute strain field from displacement field thanks to a custom method for large strain"""
          self.strain_xx = self.disp_x.copy(); self.strain_xx.fill(np.NAN)
          self.strain_xy = self.disp_x.copy(); self.strain_xy.fill(np.NAN)
          self.strain_yy = self.disp_x.copy(); self.strain_yy.fill(np.NAN)
          self.strain_yx = self.disp_x.copy(); self.strain_yx.fill(np.NAN)

          dx = self.grid_x[1][0] - self.grid_x[0][0]
          dy = self.grid_y[0][1] - self.grid_y[0][0]

          for i in range(self.size_x):
            for j in range(self.size_y):
                 du_dx = 0.
                 dv_dy = 0.
                 du_dy = 0.
                 dv_dx = 0.

                 if i-1 >= 0 and i+1 < self.size_x:
                      du = self.disp_x[i+1,j] - self.disp_x[i-1,j]
                      du_dx = du/(2.*dx)
                      du_dy = du/(2.*dy)

                 if j-1 >= 0 and j+1 < self.size_y:
                      dv = self.disp_y[i,j+1] - self.disp_y[i,j-1]
                      dv_dx = dv/(2.*dx)
                      dv_dy = dv/(2.*dy)

                 self.strain_xx[i,j] = du_dx + .5*(du_dx**2 + dv_dy**2)
                 self.strain_yy[i,j] = dv_dy + .5*(du_dx**2 + dv_dy**2)
                 self.strain_xy[i,j] = .5*(du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)

     def average(self, value, x_range, y_range):
          """Get the average value in the specified x,y range of the given field"""
          val = []
          for x in x_range:
               for y in y_range:
                    if np.isnan(value[x,y]) == False:
                         val.append(value[x,y])
          return np.average(val)

     def std(self, value, x_range, y_range):
          """Get the standard deviation value in the specified x,y range of the given field"""
          val = []
          for x in x_range:
               for y in y_range:
                    if np.isnan(value[x,y]) == False:
                         val.append(value[x,y])
          return np.std(val)

     # added by Fan
     def save_overlay(self, field, title):
         "draw and save overlaid image"
         name = self.prepare_saved_file('overlaid', 'png')
         image = cv2.imread(self.image)
         new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         Plot_and_save(new_image, self, field, name, title)



def build_grid(area, num_point, *args, **kwargs):
    xmin = area[0][0]; xmax = area[1][0]; dx = xmax - xmin
    ymin = area[0][1]; ymax = area[1][1]; dy = ymax - ymin
    point_surface = dx*dy/num_point; point_line = math.sqrt(point_surface)
    ratio = 1. if not 'ratio' in kwargs else kwargs['ratio']
    num_x = int(ratio*dx/point_line) + 1
    num_y = int(ratio*dy/point_line) + 1
    grid_x, grid_y = np.mgrid[xmin:xmax:num_x*1j, ymin:ymax:num_y*1j]
    return grid(grid_x, grid_y, num_x, num_y)

def draw_opencv(image, *args, **kwargs):
    """A function with a lot of named argument to draw opencv image
 - 'point' arg must be an array of (x,y) point
 - 'p_color' arg to choose the color of point in (r,g,b) format
 - 'pointf' to draw lines between point and pointf, pointf
   must be an array of same lenght than the point array
 - 'l_color' to choose the color of lines
 - 'grid' to display a grid, the grid must be a grid object
 - 'gr_color' to choose the grid color"""
    if type(image) == str :
         image = cv2.imread(image,0) # image = cv2.imread(image, 0)

    if 'text' in kwargs:
         text = kwargs['text']
         image = cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),4)


    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if  'point' in kwargs:
        p_color = (0, 0, 0) if not 'p_color' in kwargs else kwargs['p_color']
        for pt in kwargs['point']:
            if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                 x = int(pt[0])
                 y = int(pt[1])
                 frame = cv2.circle(frame, (x, y), 0, p_color, -1)

    scale = 1. if not 'scale' in kwargs else kwargs['scale']
    if 'pointf' in kwargs and 'point' in kwargs:
        assert len(kwargs['point']) == len(kwargs['pointf']), 'bad size'
        l_color = (255, 120, 255) if not 'l_color' in kwargs else kwargs['l_color']
        for i, pt0 in enumerate(kwargs['point']):
            pt1 = kwargs['pointf'][i]
            if np.isnan(pt0[0])==False and np.isnan(pt0[1])==False and np.isnan(pt1[0])==False and np.isnan(pt1[1])==False :
                 disp_x = (pt1[0]-pt0[0])*scale
                 disp_y = (pt1[1]-pt0[1])*scale
                 frame = cv2.arrowedLine(frame, (int(pt0[0]), int(pt0[1])), (int(pt0[0]+disp_x), int(pt0[1]+disp_y)), l_color, 1)#, tipLength=0.3)

    if 'grid' in kwargs:
        gr =  kwargs['grid']
        gr_color = (255, 255, 255) if not 'gr_color' in kwargs else kwargs['gr_color']
        for i in range(gr.size_x):
            for j in range(gr.size_y):
                 if (not math.isnan(gr.grid_x[i,j]) and
                     not math.isnan(gr.grid_y[i,j]) and
                     not math.isnan(gr.disp_x[i,j]) and
                     not math.isnan(gr.disp_y[i,j])):
                      x = int(gr.grid_x[i,j]) + int(gr.disp_x[i,j]*scale)
                      y = int(gr.grid_y[i,j]) + int(gr.disp_y[i,j]*scale)

                      if i < (gr.size_x-1):
                           if (not math.isnan(gr.grid_x[i+1,j]) and
                               not math.isnan(gr.grid_y[i+1,j]) and
                               not math.isnan(gr.disp_x[i+1,j]) and
                               not math.isnan(gr.disp_y[i+1,j])):
                                x1 = int(gr.grid_x[i+1,j]) + int(gr.disp_x[i+1,j]*scale)
                                y1 = int(gr.grid_y[i+1,j]) + int(gr.disp_y[i+1,j]*scale)
                                frame = cv2.line(frame, (x, y), (x1, y1), gr_color, 2)

                      if j < (gr.size_y-1):
                           if (not math.isnan(gr.grid_x[i,j+1]) and
                               not math.isnan(gr.grid_y[i,j+1]) and
                               not math.isnan(gr.disp_x[i,j+1]) and
                               not math.isnan(gr.disp_y[i,j+1])):
                                x1 = int(gr.grid_x[i,j+1]) + int(gr.disp_x[i,j+1]*scale)
                                y1 = int(gr.grid_y[i,j+1]) + int(gr.disp_y[i,j+1]*scale)
                                frame = cv2.line(frame, (x, y), (x1, y1), gr_color, 4)
    if 'filename' in kwargs:
         cv2.imwrite( kwargs['filename'], frame)
         return

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', frame.shape[1], frame.shape[0])
    cv2.imshow('image',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_result(result_file, image, points):
     result_file.write(image + '\t')
     for p in points:
          result_file.write(str(p[0]) + ',' + str(p[1]) + '\t')
     result_file.write('\n')

def init(image_pattern, win_size_px, grid_size_px, result_file, area_of_intersest=None, *args, **kwargs):
     """the init function is a simple wrapper function that allows to parse a
sequence of images. The displacements are computed and a result file is written
 - the first arg 'image_pattern' is the path where your image are located
 - the second arg 'win_size_px' is the size in pixel of your correlation windows
 - the third arg 'grid_size_px' is the size of your correlation grid
 - the fourth arg 'result_file' locates your result file
 - the optional argument 'area_of_intersest'gives the area of interset in (size_x,size_y) format.
   if you don't give this argument, a windows with the first image is displayed.
   You can pick in this picture manually your area of intersest.
 - you can use the named argument 'unstructured_grid=(val1,val2)' to let the 'goodFeaturesToTrack'
   opencv2 algorithm. Note that you can't use the 'spline' or the 'raw' interpolation method."""


     img_list = sorted(glob.glob(image_pattern))
     assert len(img_list) > 1, "there is not image in " + str(image_pattern)
     img_ref = cv2.imread(img_list[0], 0)
     y_1,x_1 = img_ref.shape #get the shape of image, and the greatest y and x value

     if (area_of_intersest == 'all'):
         area_of_intersest = [(0,0),(x_1,y_1)]

     # choose area of interset
     elif (area_of_intersest is None):
          print("please pick your area of intersest on the picture")
          area_of_intersest = pick_area_of_interest(img_ref)
     else:
          area_of_intersest = area_of_intersest

     # init correlation grid
     area     = area_of_intersest

     points   = []
     points_x = np.float64(np.arange(area[0][0], area[1][0], grid_size_px[0]))
     points_y = np.float64(np.arange(area[0][1], area[1][1], grid_size_px[1]))

     if 'unstructured_grid' in kwargs:
          block_size, min_dist = kwargs['unstructured_grid']
          feature_params = dict( maxCorners = 50000,
                                 qualityLevel = 0.01,
                                 minDistance = min_dist,
                                 blockSize = block_size)
          points = cv2.goodFeaturesToTrack(img_ref, mask = None, **feature_params)[:,0]
     elif 'deep_flow' in kwargs:
          points_x = np.float64(np.arange(area[0][0], area[1][0], 1))
          points_y = np.float64(np.arange(area[0][1], area[1][1], 1))
          for x in points_x:
               for y in points_y:
                    points.append(np.array([np.float32(x),np.float32(y)]))
          points = np.array(points)
     else:
          for x in points_x:
               for y in points_y:
                    points.append(np.array([np.float32(x),np.float32(y)]))
          points = np.array(points)


     # ok, display
     points_in = remove_point_outside(points, area, shape='box')


     img_ref = cv2.imread(img_list[0], 0)
     img_ref = cv2.putText(img_ref, "Displaying markers... Press any buttons to continue", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)

     #draw_opencv(img_ref, point=points_in)

     # compute grid and save it in result file
     f = open(result_file, 'w')
     xmin = points_x[0]; xmax = points_x[-1]; xnum = len(points_x)
     ymin = points_y[0]; ymax = points_y[-1]; ynum = len(points_y)
     f.write(str(xmin) + '\t' + str(xmax) + '\t' + str(int(xnum)) + '\t' + str(int(win_size_px[0])) + '\n')
     f.write(str(ymin) + '\t' + str(ymax) + '\t' + str(int(ynum)) + '\t' + str(int(win_size_px[1])) + '\n')

     # param for correlation
     lk_params = dict( winSize  = win_size_px, maxLevel = 10,
                       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

     # parse all files and write results file
     point_to_process = points_in
     write_result(f, img_list[0], point_to_process)
     for i in range(len(img_list)-1):
          print('reading image {} / {} : "{}"'.format(i+1, len(img_list), img_list[i+1]))
          #image_ref = cv2.imread(img_list[i], 0)
          image_ref = cv2.imread(img_list[0], 0)
          image_str = cv2.imread(img_list[i+1], 0)

          if 'deep_flow' in kwargs:
               winsize_x = win_size_px[0]
               final_point = cv2.calcOpticalFlowFarneback(image_ref, image_str, None, 0.5, 3, winsize_x,
                                                          10, 5, 1.2, 0) #Optical Flow
               # prev, next, flow, pyr_scale, levels, winsize, iterations,poly_n, poly_sigma
               index = 0
               ii_max = final_point.shape[0]
               jj_max = final_point.shape[1]

               for jj in range(jj_max):
                   for ii in range(ii_max):
                      #area     = [(0,0),(img_ref.shape[1],img_ref.shape[0])]
                      if (jj >= area[0][0] and jj < area[1][0] and
                          ii >= area[0][1] and ii < area[1][1]):
                          point_to_process[index] += final_point[ii,jj]
                          index += 1


          else:
               #draw_opencv(image_ref, point=points_in, pointf=final_point, l_color=(0,255,0), p_color=(0,255,0))
               final_point, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_str, points_in, None, **lk_params)
               point_to_process = final_point
          write_result(f, img_list[i+1], point_to_process)
     f.write('\n')
     f.close()






def read_dic_file(result_file, *args, **kwargs):
     """the read_dic_file is a simple wrapper function that allows to parse a dic
file (given by the init() function) and compute the strain fields. The displacement fields
can be smoothed thanks to many interpolation methods. A good interpolation method to do this
job is the 'spline' method. After this process, note that a new folder named 'pydic' is
created into the image directory where different results files are written.

These results are :
- 'disp' that contains images where the displacement of the correlation windows are highlighted. You
  can apply a scale to amplify these displacements.
- 'grid' that contains images where the correlation grid is highlighted. You
  can apply a scale to amplify the strain of this grid.
- 'marker' that contains images  where the displacement of corraleted markers are highlighted
- 'result' where you can find raw text file (csv format) that constain the computed displacement
  and strain fields of each picture.

* required argument:
  - the first arg 'result_file' must be a result file given by the init() function
* optional named arguments ;
 - 'interpolation' the allowed vals are 'raw', 'spline', 'linear', 'delaunnay', 'cubic', etc...
   a good value is 'raw' (for no interpolation) or spline that smooth your data.
 - 'save_image ' is True or False. Here you can choose if you want to save the 'disp', 'grid' and
   'marker' result images
 - 'scale_disp' is the scale (a float) that allows to amplify the displacement of the 'disp' images
 - 'scale_grid' is the scale (a float) that allows to amplify the 'grid' images
 - 'meta_info_file' is the path to a meta info file. A meta info file is a simple csv file
   that contains some additional data for each pictures such as time or load values.
"""
     # treat optional args
     interpolation = 'raw' if not 'interpolation' in kwargs else kwargs['interpolation']
     save_image    = True if not 'save_image' in kwargs else kwargs['save_image']
     scale_disp    = 1 if not 'scale_disp' in kwargs else float(kwargs['scale_disp'])
     scale_grid    = 25. if not 'scale_grid' in kwargs else float(kwargs['scale_grid'])

     # read meta info file
     meta_info = {}
     if 'meta_info_file' in kwargs:
          print('read meta info file', kwargs['meta_info_file'], '...')
          with open(kwargs['meta_info_file']) as f:
               lines = f.readlines()
               header = lines[0]
               field = header.split()
               for l in lines[1:-1]:
                    val = l.split()
                    if len(val) > 1:
                         dictionary = dict(zip(field, val))
                         meta_info[val[0]] = dictionary


     # first read grid
     with open(result_file) as f:
          head = f.readlines()[0:2]
     (xmin, xmax, xnum, win_size_x) = [float(x) for x in head[0].split()]
     (ymin, ymax, ynum, win_size_y) = [float(x) for x in head[1].split()]
     win_size = (win_size_x, win_size_y)

     grid_x, grid_y = np.mgrid[xmin:xmax:int(xnum)*1j, ymin:ymax:int(ynum)*1j]
     mygrid = grid(grid_x, grid_y, int(xnum), int(ynum))

     grid_list =[]
     # the results
     point_list = []
     image_list = []
     disp_list = []

     # parse the result file
     with open(result_file) as f:
          res = f.readlines()[2:-1]
          for line in res:
               val = line.split('\t')
               image_list.append(val[0])
               point = []
               for pair in val[1:-1]:
                    (x,y) = [float(x) for x in pair.split(',')]
                    point.append(np.array([np.float32(x),np.float32(y)]))
               point_list.append(np.array(point))
               grid_list.append(copy.deepcopy(mygrid))
     f.close()

     # compute displacement and strain
     for i, mygrid in enumerate(grid_list):
          print("\ncompute displacement and strain field of", image_list[i], "...")
          disp = compute_disp_and_remove_rigid_transform(point_list[i], point_list[0])
          mygrid.add_raw_data(win_size, image_list[0], image_list[i], point_list[0], point_list[i], disp)

          disp_list.append(disp)
          mygrid.interpolate_displacement(point_list[0], disp, method=interpolation)
          ##mygrid.compute_strain_field_DA()

          # write image files
          if (save_image):
               ##mygrid.draw_marker_img()
               mygrid.draw_disp_img(scale_disp)
               ##mygrid.draw_grid_img(scale_grid)
               if win_size_x == 1 and win_size_y == 1 :
                    mygrid.draw_disp_hsv_img()
#########################################
          # write result file
          if i>0:
               mygrid.write_result()
          if (save_image):
               # save overlaid images
               print('saving', image_list[i])
               mygrid.save_overlay(mygrid.disp_xy_indi, 'displacement')

          # add meta info to grid if it exists
          if (len(meta_info) > 0):
               img = os.path.basename(mygrid.image)
               #if not meta_info.has_key(img):
               if img not in meta_info.keys():
                    print("warning, can't affect meta deta for image", img)
               else:
                    mygrid.add_meta_info(meta_info.get(img))
                    print('add meta info', meta_info.get(img))

     # delete the first file in result folder which contains bunch of zeros
     # result_files = glob.glob(r'./pydic/result' + r'\*.csv')
     # os.remove(result_files[0])


     #average csv files
     ##print('averaging all csv files...')
     ##files = glob.glob('./img/pydic/result/*.csv')
     ##avg_dat = pd.DataFrame()
     ##count = 0
     ##for file in files:
         #avg_dat = avg_dat.add(pd.read_csv(file, delim_whitespace=True), fill_value=0) #for dat file generated by DISC
         ##avg_dat = avg_dat.add(pd.read_csv(file), fill_value=0) #for csv file generated by pydic
         ##count += 1
     ##avg_dat /= count
     ##avg_dat.to_csv('./img/pydic/result/avg.csv', index=False) #change the name of the file

     #creat a new column which contains disp_xy
     ##print('adding disp_xy column...')
     ##files = glob.glob('./img/pydic/result/avg.csv')
     ##for file in files:
         ##df = pd.read_csv(file)
         ##df["disp_xy"] = df.apply(lambda x: (x["disp_x"]**2 + x["disp_y"]**2)**0.5,axis=1)
         ##df.to_csv(file, index=False)
     print('Done!')




def compute_displacement(point, pointf):
    """To compute a displacement between two point arrays"""
    assert len(point)==len(pointf)
    values = []
    for i, pt0 in enumerate(point):
        pt1 = pointf[i]
        values.append((pt1[0]-pt0[0], pt1[1]-pt0[1]))
    return values

area = []
cropping = False

def pick_area_of_interest(PreliminaryImage):
    global area, cropping
    image = cv2.putText(PreliminaryImage, "Pick the area of interest (left click + move mouse) and press 'c' button to continue", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)

    def click_and_crop(event, x, y, flags, param):
        global area, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            area = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            area.append((x, y))
            cropping = False

            # draw a rectangle around the region of interest
            Newimage = cv2.rectangle(image, area[0], area[1], (0, 255, 0), 2)
            cv2.imshow('image', Newimage)


    clone = image.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', image.shape[1], image.shape[0])
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

            # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    print(area)###########
    return area

def remove_point_outside(points, area,  *args, **kwargs):
     shape = 'box' if not 'shape' in kwargs else kwargs['shape']
     xmin = area[0][0]; xmax = area[1][0]
     ymin = area[0][1]; ymax = area[1][1]
     res = []
     for p in points:
          x = p[0]; y = p[1]
          if ((x >= xmin) and (x <= xmax) and (y >= ymin) and (y <= ymax)):
               res.append(p)
     return np.array(res)


def compute_disp_and_remove_rigid_transform(p1, p2):
     A = []
     B = []
     removed_indices = []
     for i in range(len(p1)):
          if np.isnan(p1[i][0]):
               assert np.isnan(p1[i][0]) and np.isnan(p1[i][1]) and np.isnan(p2[i][0]) and np.isnan(p2[i][1])
               removed_indices.append(i)
          else:
               A.append(p1[i])
               B.append(p2[i])



     A = np.matrix(A)
     B =  np.matrix(B)
     assert len(A) == len(B)
     N = A.shape[0]; # total points

     centroid_A = np.mean(A, axis=0)
     centroid_B = np.mean(B, axis=0)

     # centre the points
     AA = np.matrix(A - np.tile(centroid_A, (N, 1)))
     BB = np.matrix(B - np.tile(centroid_B, (N, 1)))

     # dot is matrix multiplication for array
     H = np.transpose(AA) * BB
     U, S, Vt = np.linalg.svd(H)
     R = Vt.T * U.T

     # special reflection case
     if np.linalg.det(R) < 0:
          print("Reflection detected")
          Vt[2,:] *= -1
          R = Vt.T * U.T

     n = len(A)
     T = -R*centroid_A.T + centroid_B.T
     A2 = (R*A.T) + np.tile(T, (1, n))
     A2 = np.array(A2.T)
     out = []
     j = 0
     for i in range(len(p1)):
          if np.isnan(p1[i][0]):
               out.append(p1[i])
          else:
               out.append(A2[j])
               j = j + 1
     out = np.array(out)
     return compute_displacement(p2, out)



class Plot:
    def __init__(self, image, grid, data, title):
        self.data = np.ma.masked_invalid(data)
        self.data_copy = np.copy(self.data)
        self.grid_x = grid.grid_x
        self.grid_y = grid.grid_y
        self.data = np.ma.array(self.data, mask=self.data == np.nan)
        #need to add a mask file

        self.title = title
        self.image = image

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.25, bottom=0.25)

        self.ax.imshow(image, cmap=plt.cm.binary)
        #ax.contour(grid_x, grid_y, g, 10, linewidths=0.5, colors='k', alpha=0.7)


        self.im = self.ax.contourf(grid.grid_x, grid.grid_y, self.data, 10, cmap=plt.cm.rainbow,
                         vmax=self.data.max(), vmin=self.data.min(), alpha=0.7)
        self.contour_axis = plt.gca()

        self.ax.set_title(title)
        self.cb = self.fig.colorbar(self.im)

        axmin = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        axmax = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])
        self.smin = Slider(axmin, 'set min value', self.data.min(), self.data.max(), valinit=self.data.min(),valfmt='%1.6f')
        self.smax = Slider(axmax, 'set max value', self.data.min(), self.data.max(), valinit=self.data.max(),valfmt='%1.6f')

        self.smax.on_changed(self.update)
        self.smin.on_changed(self.update)


    def update(self, val):
        self.contour_axis.clear()
        self.ax.imshow(self.image, cmap=plt.cm.binary)
        self.data = np.copy(self.data_copy)
        self.data = np.ma.masked_where(self.data > self.smax.val, self.data)
        self.data = np.ma.masked_where(self.data < self.smin.val, self.data)
        self.data = np.ma.masked_invalid(self.data)

        self.im = self.contour_axis.contourf(self.grid_x, self.grid_y, self.data, 10, cmap=plt.cm.rainbow, alpha=0.7)


        self.cb.update_bruteforce(self.im)
        self.cb.set_clim(self.smin.val, self.smax.val)
        self.cb.set_ticks(np.linspace(self.smin.val, self.smax.val, num=10))


        # # self.cb = self.figure.colorbar(self.im)


        # self.cb.set_clim(self.smin.val, self.smax.val)
        # self.cb.on_mappable_changed(self.im)
        # self.cb.draw_all()
        # self.cb.update_normal(self.im)
        # self.cb.update_bruteforce(self.im)
        #plt.colorbar(self.im)


class Plot_and_save:
    def __init__(self, image, grid, data, name, title):
        self.data = np.ma.masked_invalid(data)
        self.data_copy = np.copy(self.data)
        self.grid_x = grid.grid_x
        self.grid_y = grid.grid_y
        self.data = np.ma.array(self.data, mask=self.data == np.nan)
        #need to add a mask file

        self.title = title
        self.image = image
        self.name = name

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        #self.fig.subplots_adjust(left=0.25, bottom=0.25)
        # self.fig.subplots_adjust(left=0, bottom=0)

        self.ax.imshow(image, cmap=plt.cm.binary)
        #ax.contour(grid_x, grid_y, g, 10, linewidths=0.5, colors='k', alpha=0.7)


        self.im = self.ax.contourf(grid.grid_x, grid.grid_y, self.data, 20, cmap=plt.cm.jet, vmax=self.data.max(), vmin=self.data.min(), alpha=0.3) #alpha=0.3
        #above is auto scale, below is fixed scale.
        scale = [i for i in range(0,16)]
        # self.im = self.ax.contourf(grid.grid_x, grid.grid_y, self.data,scale, cmap=plt.cm.jet, vmax=15, vmin=0, alpha=0.3)
        self.contour_axis = plt.gca()

        #self.ax.set_title(title)
        self.cb = self.fig.colorbar(self.im)

        #save image
        ##plt.savefig(name)
        plt.savefig(name, dpi = 600)
