FROM continuumio/miniconda3

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels defaults \
    && conda config --add channels conda-forge \
    && conda create -n drGAT python=3.10 pip=24.0 -y

RUN /bin/bash -c "source activate drGAT \
    && pip install matplotlib==3.9.0 numpy==1.26.4 pandas==2.2.2 torch==2.3.0 torch-geometric==2.5.3"

ENTRYPOINT [ "/bin/bash", "-c", "source activate drGAT && bash" ]
