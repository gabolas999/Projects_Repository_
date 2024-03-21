# 

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
import scipy.interpolate as interpolate
import os
import pandas as pd


# Function definitions

def sCale(mesh, vertices_np, X_intp, Y_intp, Z_intp):

    vertices_x, vertices_y, vertices_z = vertices_np[:,0], vertices_np[:,1], vertices_np[:,2]

    Ob_xMax = max(vertices_x)
    Ob_yMax = max(vertices_y)
    Ob_zMax = max(vertices_z)
    Ob_xMin = min(vertices_x)
    Ob_yMin = min(vertices_y)
    Ob_zMin = min(vertices_z)

    Ob_X_len = Ob_xMax - Ob_xMin  
    Ob_Y_len = Ob_yMax - Ob_yMin  
    Ob_Z_len = Ob_zMax - Ob_zMin  

    mesh_t = mesh.translate([-Ob_xMin,-Ob_yMin,-Ob_zMin], inplace=True)

    maxvector_xc = np.max(X_intp)
    maxvector_yc = np.max(Y_intp)
    maxvector_zc = np.max(Z_intp)

    ratioX = Ob_X_len / maxvector_xc
    ratioY = Ob_Y_len / maxvector_yc
    ratioZ = Ob_Z_len / maxvector_zc

    X_sc = X_intp*ratioX
    Y_sc = Y_intp*ratioY
    Z_sc = Z_intp*ratioZ

    return X_sc, Y_sc, Z_sc, mesh_t

def inTerPolation(XC, YC, ZC, ud, vd, wd, new_resolution):

    #Original sizes of the vectorfield
    xsize = int(XC.max())
    ysize = int(YC.max())
    zsize = int(ZC.max())

    #All points in the original field
    xi = np.linspace(XC.min(), XC.max(), xsize)
    yi = np.linspace(YC.min(), YC.max(), ysize)
    zi = np.linspace(ZC.min(), ZC.max(), zsize)

    #original grid
    X1, Y1, Z1= np.meshgrid(xi, yi, zi, indexing = 'ij')

    #Assigning all vectorvalues to the field
    print("Starting interpolation...")
    U = interpolate.griddata((XC, YC, ZC), ud, (X1, Y1, Z1), method='nearest')
    V = interpolate.griddata((XC, YC, ZC), vd, (X1, Y1, Z1), method='nearest')
    W = interpolate.griddata((XC, YC, ZC), wd, (X1, Y1, Z1), method='nearest')
    print("Finished")

    #Some definitions for the interpolation
    orig_grid = (xi, yi, zi)
    vectors = (U, V, W)

    #The new resolution you want in the order (X, Y, Z)
    step_x = xsize / new_resolution[0] 
    step_y = ysize / new_resolution[1]  
    step_z = zsize / new_resolution[2]  


    #New grid creation
    new_X, new_Y, new_Z = np.meshgrid(np.arange(X1.min(), X1.max() + 1, step_x),     #update this later
                                        np.arange(Y1.min(), Y1.max() + 1, step_y),
                                        np.arange(Z1.min(), Z1.max() + 1, step_z))

    new_grid = (new_X, new_Y, new_Z)

    #Interpolating function
    interpolators = [interpolate.RegularGridInterpolator(orig_grid, vectors[i], method='linear', bounds_error=False, fill_value=None) for i in range(3)]

    #Creating the new vectors
    new_vectors = [interpolators[i](new_grid) for i in range(3)]

    #The new U, V and W values in the correct shape
    new_U = new_vectors[0]
    new_V = new_vectors[1]
    new_W = new_vectors[2]

    new_X = new_X.flatten()
    new_Y = new_Y.flatten()
    new_Z = new_Z.flatten()
    new_U = new_U.flatten()
    new_V = new_V.flatten()
    new_W = new_W.flatten()

    return new_X, new_Y, new_Z, new_U, new_V, new_W

def inTersection(X_sc, Y_sc, Z_sc, mesh_t):
    x_coord = X_sc.reshape(-1,1)
    y_coord = Y_sc.reshape(-1,1)
    z_coord = Z_sc.reshape(-1,1)
    Coord_array = np.hstack((x_coord,y_coord,z_coord))

    Coord_poly = pv.PolyData(Coord_array)
    print("Checking intersecting points...")
    select = Coord_poly.select_enclosed_points(mesh_t)
    inside = select.threshold(0.5)

    inside_points_poly = inside.GetPoints().GetData()
    inside_points_np = np.array(inside_points_poly)

    Coord_array_string:np.array = np.array([",".join(item) for item in Coord_array.astype(str)])
    inside_points_array_string:np.array = np.array([",".join(item) for item in inside_points_np.astype(str)])

    #Compare the two
    density:np.array = np.isin(Coord_array_string, inside_points_array_string)*1    # the *1 changes from Booleans to binary

    return density

def geTOneslice(num_slice, X_sc, Y_sc, U_intp, V_intp, density, z_def):
    step = z_def
    X_sc_slc = X_sc[num_slice::step]
    Y_sc_slc = Y_sc[num_slice::step]
    U_intp_slc = U_intp[num_slice::step]
    V_intp_slc = V_intp[num_slice::step]
    density_slc = density[num_slice::step]

    return X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc

def eXportcsv(X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc, path_to_data, proceed=False, csvfilename='outputCSV.csv', header= ("X, Y, u, v, rho       #Comment")):
    test = 0
    if proceed:
        data = np.column_stack((X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc))
        np.savetxt(path_to_data + csvfilename, data, delimiter=',', header=header, comments='')

def eXportcsv_allslices(X_sc, Y_sc, U_intp, V_intp, density, z_def, path_to_Entire_layers_folder, proceed, header= ("X, Y, u, v, rho       #Comment")):
    num_slice = 0
    if proceed:
        for i in range(z_def):
            X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc = geTOneslice(num_slice, X_sc, Y_sc, U_intp, V_intp, density, z_def)   
            data = np.column_stack((X_sc_slc, Y_sc_slc, U_intp_slc, V_intp_slc, density_slc))
            layercsvfilename = f"/Layer_created_num_{num_slice+1}.csv"
            layerheader= header
            np.savetxt(path_to_Entire_layers_folder + layercsvfilename, data, delimiter=',', header=layerheader, comments='')
            print(f"slice number {num_slice+1} succesfully saved")
            num_slice += 1

def plotter(mesh):
        p = pv.Plotter()
        p.set_background(color = "w")
        p.add_mesh(mesh)
        p.show_bounds(color="k")
        p.show()

# Process followed in developing and understanding the code is shown below
if __name__ == '__main__':

    # SETTING UP AND LOADING DATA ------------------------------------------------------------------------------------
    #Setting up paths
    data_dir = os.path.dirname(__file__)
    path_to_data = data_dir.replace("\Interpolation_Intersection_Slicing", "") + "\Data_2"

    # Load the OBJ file
    mesh = pv.read(path_to_data + "\Design_GE_Bracket.obj")

    # Load Vector_field_to_edit_stuff.csv 
    df = pd.read_csv(path_to_data + "\Vector_Field_Input.csv")  
    vector_df = df.to_numpy()


    # OBJECT HANDLING -----------------------------------------------------------------------------------------------
    # Plot and visualize the initial mesh
    plot_initial_mesh = True
    if plot_initial_mesh:
        p = pv.Plotter()
        p.set_background(color = "w")
        p.add_mesh(mesh)
        p.show_bounds(color="k")
        p.show()

    # Get vertices
    vertices = mesh.points
    vertices_np = np.array(vertices.copy())

    # Getting 1D arrays each containing the x, y, z coordinates of points and identify boundaries
    vertices_x, vertices_y, vertices_z = vertices_np[:,0], vertices_np[:,1], vertices_np[:,2]
    Ob_xMax = max(vertices_x)
    Ob_yMax = max(vertices_y)
    Ob_zMax = max(vertices_z)
    Ob_xMin = min(vertices_x)
    Ob_yMin = min(vertices_y)
    Ob_zMin = min(vertices_z)

    # Lengths of object
    Ob_X_len = Ob_xMax - Ob_xMin  #172.84087
    Ob_Y_len = Ob_yMax - Ob_yMin  #103.78857
    Ob_Z_len = Ob_zMax - Ob_zMin  #60.3064

    # Scale and translate the mesh
    mesh = mesh.translate([-Ob_xMin,-Ob_yMin,-Ob_zMin], inplace=True)
    vertices_t = mesh.points
    vertices_t_np = np.array(vertices_t.copy())

    # Getting 1D arrays each containing the x_t, y_t, z_t coordinates of translated points and identify boundaries
    vertices_x_t, vertices_y_t, vertices_z_t  = vertices_t_np[:,0], vertices_t_np[:,1], vertices_t_np[:,2]
    Ob_xMax_t = max(vertices_x_t)
    Ob_yMax_t = max(vertices_y_t)
    Ob_zMax_t = max(vertices_z_t)
    Ob_xMin_t = min(vertices_x_t)
    Ob_yMin_t = min(vertices_y_t)
    Ob_zMin_t = min(vertices_z_t)





    # VECTOR FIELD HANDLING ------------------------------------------------------------------------------------------------
    vector_xc = vector_df[:,0]
    vector_yc = vector_df[:,1]
    vector_zc = vector_df[:,2]
    vector_ud = vector_df[:,3]
    vector_vd = vector_df[:,4]
    vector_wd = vector_df[:,5]

    # Max values
    maxvector_xc = max(vector_xc)
    maxvector_yc = max(vector_yc)
    maxvector_zc = max(vector_zc)

    # Relevant ratios for scaling 
    ratioX = Ob_X_len / maxvector_xc
    ratioY = Ob_Y_len / maxvector_yc
    ratioZ = Ob_Z_len / maxvector_zc

    # Scale the vector field coordinates to match the object
    vector_df[:,0] *= ratioX
    vector_df[:,1] *= ratioY
    vector_df[:,2] *= ratioZ

    # Recollect all values of vector field
    XC = vector_df[:,0]
    YC = vector_df[:,1]
    ZC = vector_df[:,2]
    ud = vector_df[:,3]
    vd = vector_df[:,4]
    wd = vector_df[:,5]






    # Intersection --------------------------------------------------------------------------------------------------------------

    # Obtain numpy array of floats coordinates
    x_coord = XC.reshape(-1,1)
    y_coord = YC.reshape(-1,1)
    z_coord = ZC.reshape(-1,1)
    coordinate_array = np.hstack((x_coord,y_coord,z_coord))  #shape = (397955, 3)

    # Transform it into pv.PolyData object to check whether they are inside or outside of the GE bracket object (another pv object)
    # Hence it becomes a list of points in the domain of pv
    points_poly = pv.PolyData(coordinate_array)

    # Sort the list of points depending on whether they are inside or outside of the GE bracket object 
    select = points_poly.select_enclosed_points(mesh)
    inside = select.threshold(0.5)
    outside = select.threshold(0.5, invert=True)
    inside_points = inside.GetPoints().GetData()

    # Transform list of points inside back to numpy
    inside_points_np = np.array(inside_points)  # shape = (23688, 3)

    # Plot and visualize the inner mesh
    plot_inner_mesh = True
    if plot_inner_mesh:
        p = pv.Plotter()
        p.set_background(color = "w")
        p.add_mesh(inside)
        p.show_bounds(color="k")
        p.show()



    # --------------------(In the end this approach was not really used)-------------------- #
    # Transform both numpy arrays into a list of string arrays, where each item is a string (probably not very elegant but works for now)
    coordinate_array_string:list = [",".join(item) for item in coordinate_array.astype(str)]    # length = 397955
    inside_points_lst:list = [",".join(item) for item in inside_points_np.astype(str)]          # length = 23688
    # Convert to numpy
    coordinate_array_string:np.array = np.array(coordinate_array_string)
    inside_points_lst:np.array = np.array(inside_points_lst)
    # Create the mask
    mask:np.array = np.isin(coordinate_array_string, inside_points_lst)     # shape = (397955,)   number of Trues = 23688
    # --------------------(--------------------------------------------)-------------------- #


    # Create a slice from the entire dataset
    start = 39  # Index of the slice to use
    step = 71   # Amount of z values currently present

    XC = XC[start::step]
    YC = YC[start::step]
    mask = mask[start::step]

    z = np.ones((mask.shape[0], 1))

    slice = np.column_stack((XC[mask], YC[mask], z[mask]))
    Slice_poly = pv.PolyData(slice)

    # Plot and visualize the slice
    plot_slice = True
    if plot_slice:
        p = pv.Plotter()
        p.set_background(color = "w")
        p.show_bounds(color="k")
        p.add_mesh(Slice_poly, color = "k")
        p.show()

