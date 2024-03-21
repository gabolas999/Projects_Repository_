
In this project, I Developed, Verified & Validated a Flight-Dynamics and Stability Model for a business jet. These files are part of Building the simulation and of the process of validating it.

To validate the model, data from a real flight test is used, both in form of Excel file (*PDF_08-08-2021*), which has some stationary data, and a .mat file which contains all the sensor measurements made throughout the entire flight test (*FTISxprt-20210808_143959*). The flight dynamics numeric model is made using equations of motion and derivations not justified here. I developed this code along with another colleague, working together and making decisions such that I was involved in the entire code.

As part of the bigger project, in which we had no guidance, I also performed the tasks of leading, managing and coordinating the team. So subsystem definition, organizing file dependencies, and managing task distribution. Also dealt with Git control and management, conflict resolution, and branch creation and merging. 

Regarding the files, the main.py is essentially state_space.py, which imports variables and functions from other files. It 
The inputs are Static Stability values (*SS1.csv*, and *SS2.csv*). Conversely, *SS2.csv* is created using *SS2_long.py*. 
Finally all parameters are centralized in *Cit_par24.py*, where you can select the simulation you want to run. *utilities.py* is used to centralize various functions needed throughout the different files, and in other files not included here.

Project took place: February 2024-March 2024