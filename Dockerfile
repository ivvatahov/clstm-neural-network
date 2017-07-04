FROM tensorflow/tensorflow:latest-devel-py3

WORKDIR /app/

# ADD notebooks/*.ipynb /app/notebooks/

ADD requirements.txt /app/install/

RUN cd /app/install/ && pip install -r requirements.txt

RUN pip3 install nltk 

RUN python3 -m nltk.downloader punkt

# RUN rm -rf /notebooks/*

EXPOSE 8888