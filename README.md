# Distributed-Attack-Resilient-Platooning-Against-False-Data-Injection
<p align="center">
  <img src="readme_images/Intro_pic.png" alt="jetracer_intro_pic_v2" width="850"/>
</p>

## Content
This repo contains the code and data relative to the paper "Distributed Attack-Resilient Platooning Against False Data Injection" by Lorenzo Lyons, Manuel Boldrer and Laura Ferranti.
{l.lyons,l.ferranti}@tudelft.nl boldrman@fel.cvut.cz

### Installation
A working ACADOS installation is needed for folders "1_simulations_and_tuning".
ROS packages need a working ROS installation and need to be placed in a catkin workspace. 

### 1_simulations_and_tuning
This folder contains the code used to aid the theoretical development of the proposed method. In particular the linear controller gain tuning procedure and a simulation environment showing different platooning scenarios not shown in the paper.

### 2_platooning_experiments_data_processing
This folder contains the experimental data and the code used to generate the figures in the paper. Note that the .bag files recorded during the experiments were compressed due to size issues when uploading. In particular this folder features the code used to process the experimental data as well as the *attack detector* gain tuning process.

### ROS_packages
The package "platooning_utilities" is intended to be run on a laptop and can be used while playing the provided ros .bag files in "2_platooning_experiments_data_processing/experiment_data", while platooning_pkg is intended to run on [DART](https://github.com/Lorenzo-Lyons/DART.git), the platform used to conduct the experiments. 



<h2 align="center"><strong>Distributed autonomous platoon rearranging</strong></h2>

<p align="center">
  <img src="readme_images/rearranging.gif" alt="rearranging_gif"/>
</p>
