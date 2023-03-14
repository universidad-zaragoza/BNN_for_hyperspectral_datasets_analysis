FROM python:3.8

RUN python3 -m pip install numpy
RUN python3 -m pip install matplotlib
RUN python3 -m pip install spectral
RUN python3 -m pip install scipy
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install tensorflow
RUN python3 -m pip install tensorflow_probability

RUN mkdir /workdir
WORKDIR /workdir

CMD ./launch.sh

