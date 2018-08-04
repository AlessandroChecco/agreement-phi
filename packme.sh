workon flask3
pyinstaller --onefile --hidden-import=theano.tensor.shared_randomstreams phi.py
