
In this project I performed data processing and matching as part of a larger group.
I identified the inputs and outputs required for my section of the code, and ensured it complied with what other 
members of the project needed. 
Essentially, the project involved matching data from a 3D cubic vector field, and the 3D design of an aircraft part (GE Bracket). 
The 3D vector field was created using FEM, and represents the direction of stress for a certain loading of the part.
Hence only the vectors inside the part are relevant. The next step is slicing the vector field horizontally, to get 
the "layers" a 3D printer would need to print. Interpolation in all directions of the vector field is done, to ensure 
there is enough definition of the vector field.

The final output is the interpolated vector field, where each point is appended with a density variable, depending 
whether it lies inside the part or not.



I worked in this code individually, but it is part of a larger project, in which the group of students assisted a PHD researcher. 
I was tasked with the data handling.
Project took place: March 2023-May 2024