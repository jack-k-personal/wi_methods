# Instructions

Step 1: `mkdir indexes`

To verify that the BLP is working, compared to binary search run:

```
python3 bs_vs_blp_wi_grid.py
```

Note: Binary search with N_GRID = 5 should take about 8-10 seconds



To compute a grid efficiently with BLP, run:

```
python3 blp_wi_grid.py
```

To verify that the BQP is working, compared to the BLP-computed grid, run:
```
python3 bqp_wi_max.py

```

Note: the above file should take around 6 seconds


# Setup
## Python packages needed:
- numpy
- pandas
- scipy
- gurobipy

```
pip3 install numpy pandas scipy
```

## Installing Gurobi and gurobipy
- In a web browser Register for a Gurobi account or login at https://www.gurobi.com/downloads/end-user-license-agreement-academic/ 
- Navigate to https://www.gurobi.com/downloads/ and select `Gurobi Optimizer`
- Review the EULA, then click `I accept the End User License Agreement`
- Identify the latest installation... as of writing, it is 9.1.2, and the following commands will reflect that. However, if the latest version has changed, you can replace 9.1.2 in the following commands with the newer version number and/or links on the Gurobi website.
- Navigate to `https://www.gurobi.com/wp-content/uploads/2021/04/README_9.1.2.txt` and read the README
- Back on the digial ocean server terminal: `mkdir tools`
- `cd tools`
- `wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz`
- `tar xvzf gurobi9.1.2_linux64.tar.gz`
- Add the following lines to your ~/.bashrc file, e.g., via `vim ~/.bashrc`
```
export GUROBI_HOME="/root/tools/gurobi912/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
``` 
- Run `source ~/.bashrc`
- On the browser, navigate to https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- Review the conditions, then click `I accept these conditions`
- Scroll down to **Installation** and copy the command that looks like `grbgetkey 00000000-0000-0000-0000-000000000000`, then paste and run in the server terminal window
- Enter `Y` to select the default options when prompted
- `pip3 install gurobipy`

