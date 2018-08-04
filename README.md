# Agreement measure Phi
Source code for inter-rater agreement measure Phi.

## Requirements
python 3+, pymc3 version 3.3. See requirements files for tested working versions on linux and osx.

## Example
Input is a np 2-dimensional array with NaN for missing values, every row represents a different document, every column a different worker. Note that Phi does not take in account worker bias, so the order in which ratings appear for each document does not matter. For this reasons, missing values and a sparse representation is needed only when documents have different number of ratings.

``run_phi( data_phi,limits=[0,100],keep_missing=True,fast=True,njobs=4,verbose=False,table=False,N=500)``

- limits defines the scale
- keep_missing = True [default True] if you have many NaNs you might want to switch to false
- fast = True [default True] inference technique
- N=500 [default 1000] number of iterations
- verbose = False [default False] if True it shows more information
- table = False [default False] if True more verbose output
- njobs = 4 number of parallel jobs

output example:
``{'agreement': 0.023088447111559884, 'computation_time': 58.108173847198486, 'convergence_test': True, 'interval': array([-0.03132854,  0.06889001])}``

If  convergence_test is False we recommend to increase N.

## References
If you use it for academic publications, please cite out paper
Checco, A., Roitero, K., Maddalena, E., Mizzaro, S., & Demartini, G. (2017). Let’s agree to disagree: Fixing agreement measures for crowdsourcing.
```
@article{checco2017let,
  title={Let’s agree to disagree: Fixing agreement measures for crowdsourcing},
  author={Checco, Alessandro and Roitero, Kevin and Maddalena, Eddy and Mizzaro, Stefano and Demartini, Gianluca},
  year={2017},
  publisher={AAAI Press}
}
```
