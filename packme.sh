cd agreement_phi
#python3 -m pip install --upgrade setuptools wheel
rm -rf ./dist
rm -rf ./build
rm -rf ./agreement_phi.egg-info
python3 setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
rm -rf ./dist
rm -rf ./build
rm -rf ./agreement_phi.egg-info
echo "You can install with:"
echo "pip install --index-url https://test.pypi.org/simple/ agreement-phi"