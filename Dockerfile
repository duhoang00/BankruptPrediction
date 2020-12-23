FROM heroku/miniconda

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./ /opt/
WORKDIR /opt

RUN conda install scikit-learn
RUN conda install scipy
RUN conda install imblearn
RUN conda install graphviz
RUN conda install missingno

CMD python app.py