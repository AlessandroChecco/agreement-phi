workon flask3
pyinstaller --onefile --hidden-import=theano.tensor.shared_randomstreams --hidden-import=scipy.optimize --hidden-import scipy._lib.messagestream --hidden-import pandas._libs.tslibs.timedeltas phi.py
