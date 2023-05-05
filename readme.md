# ME5406 Project 1

This is the introduction of all the script and how you should build the test environment.

## Build the environment
First, this repository is under the python 3.6.10 so please make sure you have correct version installed.

Then, please download Anaconda in your machine with this link: https://www.anaconda.com/products/distribution#download-section

After downloading, please run this to install Anaconda automatically:
        
        bash Anaconda-latest-Linux-x86_64.sh

After you installed Anaconda successfully, please run this commands in your terminal to build the environment: 

        conda create --name me5406env python=3.6.10 -y
        conda activate me5406env
        pip install numpy==1.19.2
        pip install argon2-cffi==21.3.0
        pip install jsonschema==3.2.0
        pip install gym==0.21.0

Note: If you see something error, please try to install the dependencies again, you should be fine.

## Run the script
The naming rule of all these script is actually quite intuitive:
1. mcwoes is short for Monte Carlo Control without ES
2. env4by4 is the environment of 4x4 frozen lake scene
3. env10by10 is the environment of 10x10 frozen lake scene
4. drawplot4by4 and drawplot10by10_three plot all the graphs in the report
5. drawplot10by10_sarsa_ql only plot the relationship between learning rate and successful rate
6. the script in utility can generate a 10x10 random map, then just copy this map to env10by10 to renew the map
