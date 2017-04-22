apt-get -y install virtualenv python3-dev build-essential
virtualenv -p python3 tf_playground
source tf_playground/bin/activate
pip3 install pandas google.cloud nltk jupyter bs4
pip3 install --upgrade tensorflow-gpu
jupyter notebook --generate-config

