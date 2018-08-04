# Agreement measure Phi
Source code for inter-rater agreement measure Phi.

## Requirements
python 3+, pymc3 version 3.3. See requirements files for tested working versions on linux and osx.

## Example
Input is a numpy 2-dimensional array with NaN for missing values, or equivalently a python list of lists. Every row represents a different document, every column a different rating. Note that Phi does not take in account worker bias, so the order in which ratings appear for each document does not matter. For this reasons, missing values and a sparse representation is needed only when documents have different number of ratings.

### input example 
```
m_random = np.random.randint(5, size=(5, 10)).tolist()
m_random[0][1]=np.nan
```
or equivalently
```
m_random = np.random.randint(5, size=(5, 10)).astype(float)
m_random[0][1]=np.nan
```
### running the measure inference
``run_phi( data_phi,limits=[0,100],keep_missing=True,fast=True,njobs=4,verbose=False,table=False,N=500)``

- limits defines the scale
- keep_missing = True [default True] if you have many NaNs you might want to switch to false
- fast = True [default True] inference technique
- N=500 [default 1000] number of iterations
- verbose = False [default False] if True it shows more information
- table = False [default False] if True more verbose output
- njobs = 4 number of parallel jobs
Note that the code will try to infer the limits of the scale, but it's highly suggested to include them (in case some elements on the boundary are missing). For this example the parameter limits would be ``limits=[0,4]``.

### output example
```
{'agreement': 0.023088447111559884, 'computation_time': 58.108173847198486, 'convergence_test': True, '95% Highest Posterior Density interval': array([-0.03132854,  0.06889001])}
```

If  convergence_test is False we recommend to increase N.

## References
If you use it for academic publications, please cite out paper:

Checco, A., Roitero, A., Maddalena, E., Mizzaro, S., & Demartini, G. (2017). Let’s Agree to Disagree: Fixing Agreement Measures for Crowdsourcing. In Proceedings of the Fifth AAAI Conference on Human Computation and Crowdsourcing (HCOMP-17) (pp. 11-20). AAAI Press.
```
@inproceedings{checco2017let,
  title={Let’s Agree to Disagree: Fixing Agreement Measures for Crowdsourcing},
  author={Checco, A and Roitero, A and Maddalena, E and Mizzaro, S and Demartini, G},
  booktitle={Proceedings of the Fifth AAAI Conference on Human Computation and Crowdsourcing (HCOMP-17)},
  pages={11--20},
  year={2017},
  organization={AAAI Press}
}
```
