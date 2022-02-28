import os
import json

# Data Wrangling

import h5py
import numpy as np
import pandas as pd

import smallestenclosingcircle
from Load_Ethogram import diff_between_List
from Load_Ethogram import add_hierarchical_level


class TrajectoryPreprocessing:
    
    def __init__(self,
                 fish,position_column=['img_l','img_c'],
                 angle_column='body_angle_cart',
                 out_xy_style='cartesian',
                 out_xy_unit='mm',
                 out_time_unit='camera_frame',
                 ):
        
        self.fps = fish.fps
        self.width = fish.width
        self.height = fish.height
        self.pixel_size= fish.pixel_size
        info = fish.load_info()
        self.cam_center = np.array(info['cam_center'])
        P = np.array(info['cam2sh_transformation'])
        P = P.reshape(2,2)
        P = np.linalg.inv(P)
        self.cam2sh_transformation = P
        
        self.out_xy_style = out_xy_style # 'cartesian','image','shader','polar'
        self.out_xy_unit = out_xy_unit # 'mm','pix','shader'
        self.out_time_unit = out_time_unit #'camera_frame','sec'

        self.position_column = position_column
        self.angle_column = angle_column

    def get_trajectory_df(self,camdf):
        if self.out_xy_style == 'cartesian':
            position_df = self._get_xy_trajectory(camdf)
            xc,yc,radius = self._compute_outer_circle(position_df,interval=100)
            circle=(xc,yc,radius)
            position_df = self._center_xy(position_df['x'],position_df['y'],xc,yc)
        if self.out_xy_style == 'image':
            position_df = self._get_image_trajectory(camdf)
            lc,cc,radius = self._compute_outer_circle(position_df,interval=100)
            circle=(lc,cc,radius)
        if self.out_xy_style == 'shader':
            position_df = self._get_shader_trajectory(camdf)
            lc,cc,radius = self._compute_outer_circle(position_df,interval=100)
            circle=(lc,cc,radius)

        angle_df = self._get_angle_trajectory(camdf)
        trajectory_df = pd.DataFrame.merge(position_df,angle_df,how='outer',left_index=True, right_index=True)
        #outlier_df = trajectory.find_outliers(LAG_MAX,SPEED_MAX)(camlog)
        #time_df = trajectory.get_time(input_time_unit='camera_frame')(camlog)
        return trajectory_df,circle        
    
    def _get_shader_trajectory(self,df):
        l = df[self.position_column[0]]
        c = df[self.position_column[1]]
        x,y = self.__position_img2cartesian(l,c)
        x = x - self.cam_center[0]
        y = y - self.cam_center[1]
        P = (self.cam2sh_transformation)
        x = P[0,0]*x + P[0,1]*y
        y = P[1,0]*x + P[1,1]*y
        xy = { 'x': x, 'y': y } 
        return pd.DataFrame(xy) 
        
        
    def _get_image_trajectory(self,df):
        l = df[self.position_column[0]]
        c = df[self.position_column[1]]
        lc = { 'l': l, 'c': c } 
        return pd.DataFrame(lc) 
            
    def _get_xy_trajectory(self,df):
        l = df[self.position_column[0]]
        c = df[self.position_column[1]]
        x,y = self.__position_img2cartesian(l,c)
        if self.out_xy_unit=='mm':
            x = x * self.pixel_size/1000.
            y = y * self.pixel_size/1000.
        xy = { 'x': x, 'y': y } 
        return pd.DataFrame(xy) 
               
    def _get_angle_trajectory(self,df,unwrap=True):
        # Make sure the angle out is 
        in_angle=df[self.angle_column]
        si = np.sin(in_angle)
        co = np.cos(in_angle)
        angle = np.arctan2(si,co)
        if unwrap:
            angle[:] = np.unwrap(angle,axis=0)

        return pd.DataFrame({'angle':angle})

    def _center_xy(self,x,y,xc,yc):
        x_centered = x - xc
        y_centered = y - yc
        xy = { 'x': x_centered, 'y': y_centered } 
        return pd.DataFrame(xy)
    
    # Class that are not meant to be inherited: 
    def __angle_img2cartesian(self,angle):
        theta = np.pi/2-angle
        return theta
        
    def __position_img2cartesian(self,l,c):
        x = c
        y = self.height-l
        return x,y
    
    #@property
    def _compute_outer_circle(self,df,interval=100):
        p= [(df.iloc[i,0],df.iloc[i,1]) for i in np.arange(0,df.shape[0],interval)]
        Circle = smallestenclosingcircle.make_circle(p)
        xc=Circle[0]
        yc=Circle[1]
        radius=Circle[2]
        return (xc,yc,radius)    
                
