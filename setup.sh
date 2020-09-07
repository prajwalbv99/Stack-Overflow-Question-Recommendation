#!/bin/sh
# wget https://bootstrap.pypa.io/get-pip.py
sudo python3 -m pip install --upgrade pip

sudo yum install -y git
aws s3 cp s3://cloud-stack-overflow/requirements.txt .

# Install the pip requirements
sudo /usr/local/bin/pip3 install -r requirements.txt
python3 -m nltk.downloader all
