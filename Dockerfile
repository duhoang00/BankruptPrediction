FROM heroku/miniconda

# Grab requirements.txt.
ADD ./webapp/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./ /opt/
WORKDIR /opt

RUN conda install -c anaconda scikit-learn
RUN conda install -c anaconda scipy
RUN conda install -c conda-forge imbalanced-learn
RUN conda install -c anaconda graphviz
RUN conda install -c conda-forge missingno

CMD python app.py