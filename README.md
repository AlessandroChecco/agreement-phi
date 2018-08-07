# Agreement measure Phi
Source code for inter-rater agreement measure Phi. Live demo here: http://agreement-measure.sheffield.ac.uk

## Requirements
python 3+, pymc3 3.3+. See requirements files for tested working versions on linux and osx.

## Installation - with pip
Simply run ``pip install agreement_phi``.
This will provide a module and a command line executable called ``run_phi``.

## Installation - without pip
Download the folder.

## Example - from command line
Prepare a csv file (no header, each row is a document, each column a rater), leaving empty the missing values. For example ``input.csv``:
```
1,2,,3
1,1,2,
4,3,2,1
``` 
And execute from the console ``run_phi --file input.csv --limits 1 4``.
More details obtained running ``run_phi --h``:
```
usage: agreement_phi.py [-h] -f FILE [-v] [-l val val]

Phi Agreement Measure

optional arguments:
  -h, --help                     show this help message and exit
  -f FILE, --file FILE           input FILE <REQUIRED>
  -v, --verbose                  print verbose messages
  -l val val, --limits val val   Set limits <RECOMMENDED> (two values separated by a space)
```

## Example - from python
Input is a numpy 2-dimensional array with NaN for missing values, or equivalently a python list of lists (where each list is a set of ratings for a document, of the same length with nan padding as needed). Every row represents a different document, every column a different rating. Note that Phi does not take in account rater bias, so the order in which ratings appear for each document does not matter. For this reasons, missing values and a sparse representation is needed only when documents have different number of ratings.

### Input example 
```
import numpy as np
m_random = np.random.randint(5, size=(5, 10)).tolist()
m_random[0][1]=np.nan
```
or equivalently
```
m_random = np.random.randint(5, size=(5, 10)).astype(float)
m_random[0][1]=np.nan
```

### Running the measure inference
```
from agreement_phi import run_phi
run_phi(data=m_random,limits=[0,4],keep_missing=True,fast=True,njobs=4,verbose=False,table=False,N=500)
```

- ``data`` [non optional] is the matrix or list of lists of input (all lists of the same length with nan padding if needed).

#### OPTIONAL PARAMETERS:

- ``limits`` defines the scale [automatically inferred by default]. It's a list with the minimum and maximum (included) of the scale.
- ``keep_missing`` [automatically inferred by default based on number of NaNs] boolean. If you have imbalanced documents in terms of number of ratings, you might want to switch it to False,
- ``fast`` [default True] boolean. Whether to use or not the fast inferential technique (note that you might want to also change ``N`` in that case).
- ``N`` [default 1000] integer. Number of iterations. Increase it if ``convergence_test`` is False.
- ``verbose`` [default False] boolean. If True it shows more information
- ``table`` [default False] boolean. If True more verbose output in form of a table.
- ``njobs`` [default 1] integer. Number of parallel jobs. Set it equal to the number of CPUs available.
- ``binning`` [default True] boolean. If False consider the values in the boundary of scale non binned: this is useful when using a discrete scale and the value in the boundaries should be considered adhering to the limits and not in the center of the corresponding bin. This is useful when the value of the boundaries have a strong meaning (for example [absolutely not, a bit, medium, totally]) where answering in the boundary of the scale is not in a bin as close as the second step in the scale.

Note that the code will try to infer the limits of the scale, but it's highly suggested to include them (in case some elements on the boundary are missing). For the example shown above the parameter limits would be ``limits=[0,4]``.

Note that ``keep_missing`` will be automatically inferred, but for highly inbalanced datasets (per document number of ratings distribution) it can be overriden by manually setting this option.

### Output example
```
{'agreement': 0.023088447111559884, 'computation_time': 58.108173847198486, 'convergence_test': True, 'interval': array([-0.03132854,  0.06889001])}
```

Where 'interval' represents the 95% Highest Posterior Density interval.
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
