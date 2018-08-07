source `which virtualenvwrapper.sh-3.5`
workon newpymc3
cd agreement_phi
#python3 -m pip install --upgrade setuptools wheel
rm -rf ./dist
rm -rf ./build
rm -rf ./agreement_phi.egg-info
rm -rf ./__pycache__
python3 setup.py sdist bdist_wheel
twine upload dist/*
rm -rf ./dist
rm -rf ./build
rm -rf ./agreement_phi.egg-info
echo "You can install with:"
echo "pip install agreement_phi"
deactivate
