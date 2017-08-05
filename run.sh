jupyter nbconvert --to python $1.ipynb
python $1.py | tee $1.log
