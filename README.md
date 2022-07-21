### Paper
paper information comes here after its publication.

### How to run
Use `run_BO.py` to carry out BO and display the result. 

1. Example 1: method F

```$ python run_BO.py --num 100 --init 5 --run 3 --obj circle```
executes BO until having `100` observations with `5` initial observations for `3` times to average the results. 

2. Example 2: @0

```$ python run_BO.py --floor 0```

`--floor {value}` option specifies the constant value to replace failed observations. By default, floor padding trick is employed. 

3. Example 3: other objective functions

```$ python run_BO.py --obj hole``` 
runs experiment with the Hole function. 

4. Example 4: put Gaussian observation noise

```$ python run_BO.py --alpha 0.005``` 
adds a Gaussian noise to the observation with its variance being `0.005`.

5. Example 5: use binary classifier

```$ python run_BO.py --binary --obj hole ``` runs method FB. 

Add `--binary` to enable the binary classifier. Can be combined with `--floor` option. For example, `--binary --floor 0` runs B@0.

### Software version
Codes are confirmed to run with the following libraries. Likely to be compatible with newer versions. 

* `python`: `3.7.9`
* `numpy`: `1.19.2`
* `scipy`: `1.5.2`
* `sklearn`: `1.0.2`
* `torch`: `1.8.1`
* `gpytorch`: `1.5.1`
* `matplotlib`: `3.5.0`

### Files
* `README.md`: This file. 
* `LICENSE.pdf`: Document of agreement for using this sample code. Read this carefully before using the code. 
* `run_BO.py`: Script to execute BO sequence. 
* `BO_core.py`: Implements BO class. 
* `obj_func.py`: Implements objective functions. 
* `visualize.py`: Contains some functions to plot optimization results. 
* `lhsmdu.py`: Latin hypercube sampling package for acquisition function. Repository: https://dx.doi.org/10.5281/zenodo.2578780  Paper:http://dx.doi.org/10.1016%2Fj.jspi.2011.09.016