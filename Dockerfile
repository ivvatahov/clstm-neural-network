FROM tensorflow/tensorflow:nightly-py3

WORKDIR /app/

ADD *.ipynb /app/notebooks/

ADD requirements.txt /app/install/

RUN cd /app/install/ && pip install -r requirements.txt

RUN pip install nltk

RUN python -m nltk.downloader punkt

RUN rm -rf /notebooks/*

EXPOSE 8888