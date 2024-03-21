"""
File containing all the functions needed do load, scale, intersect, and slice the GE bracket and Vector field. 
- This is the only file that should be run, it imports functions defined in otherdirectionsOfficial_Interpolation_2, and Official_Slicing_Intersection_2
- The input for this part of the code is a csv file containing the vector field (Vector_Field_Input.csv), and a object file containing the GE bracket
model. (these can be found in Data_2 folder)
- The output for this part of the code is a csv file containing the interpolated vector field, appended with an extra colum which 
specifies a density (either 0 or 1) to each point if they are outside or inside of the object. (these can be found in Data_2/Entire_layers_2 folder)


This code is optimized so only the data is dealt with, no plotting is enabled
Functions were defined that clean up and shorten the process, for an in depth procedure of how the code was developed, go to Official_Slicing_intersection_2.py
"""

#import packages
import pyvista as pv
import numpy as np
import os

#import other python files
from Functions_n_codeprocess import sCale, inTerPolation, inTersection, geTOneslice, eXportcsv, eXportcsv_allslices
import pandas as pd


#INPUT FROM THE USER:
#==============================================================================================================================
#==============================================================================================================================
#What new resolution for the vector field do you want? (multiplication of the original resolution for the three axes) (must be integer)
x_mult = 1     # Initial=95
y_mult = 1     # Initial=59
z_mult = 2     # Initial=36

#Slice to obtain
num_slice = 57   #has to be in the range of the resolution in z direction between 36 and 36*z_mult

#Saving 
proceed = True
csvfilename = '/CSV_output.csv'
header= (f"X, Y, u, v, rho       # Resolution augmented in the x, y, z directions by x*{x_mult}, y*{y_mult}, z*{z_mult}")

oneslice = True  # Opposite option is all slices
#==============================================================================================================================
#==============================================================================================================================


#Setting up paths for reading data
data_dir = os.path.dirname(__file__)
path_to_data = data_dir.replace("\Interpolation_Intersection_Slicing", "") + "\Data_2"
path_to_Entire_layers_folder = path_to_data + "\Entire_layers_2"

#Load the vectorfield data
df = pd.read_csv(path_to_data + "\Vector_Field_Input.csv")
dataframe = df.to_numpy()

#Load the OBJ file
mesh = pv.read(path_to_data + "\Design_GE_Bracket.obj")

#Data handling
#Frame the vector field data
XC = dataframe[:,0]
YC = dataframe[:,1]
ZC = dataframe[:,2]
ud = dataframe[:,3]
vd = dataframe[:,4]
wd = dataframe[:,5]

#Get vertices from object
vertices = mesh.points
vertices_np = np.array(vertices.copy())

#Definition of the resolution
x_res = 95 * x_mult         # These values were hardcoded, since they are dependent on the dataset provided
y_res = 59 * y_mult
z_res = 36 * z_mult
new_resolution = (x_res, y_res, z_res)

# Interpolate values into the vector field
Interpolated_Field = inTerPolation(XC, YC, ZC, ud, vd, wd, new_resolution)
print("Interpolation Finished!")

X_intp, Y_intp, Z_intp, U_intp, V_intp, W_intp = Interpolated_Field
print("Amount of points in cube", X_intp.shape, Y_intp.shape, Z_intp.shape)

# Scale vector field to match the vector field limits to the object dimensions
X_sc, Y_sc, Z_sc, mesh_t = sCale(mesh, vertices_np, X_intp, Y_intp, Z_intp)
print("sCaled successfully!")

# Identify which points of the vector field lie inside the object
density = inTersection(X_sc, Y_sc, Z_sc, mesh_t)    # Longest computational time
print("inTersected successfully!")

# Slice the object along the x-y plane, at the z value selected (num_slice) 
X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc = geTOneslice(num_slice, X_sc, Y_sc, U_intp, V_intp, density, z_def=new_resolution[2])
print("geTOneslice successful")
print("Amount of points in one slice", X_sc_slc.shape, Y_sc_slc.shape)

if oneslice:
    # Export the slice into a csv
    eXportcsv(X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc, path_to_data, proceed, csvfilename, header)
    print(f"eXporting csv successful: {csvfilename}")
else:
    # Get all the slices and export all into separate folder
    z_def=new_resolution[2]
    eXportcsv_allslices(X_sc, Y_sc, U_intp, V_intp, density, z_def, path_to_Entire_layers_folder, proceed=True, header=header)