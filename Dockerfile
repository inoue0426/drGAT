FROM continuumio/miniconda3

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

COPY . /app

# コンテナが起動時にdrGAT環境をアクティベートするようにする
ENTRYPOINT [ "/bin/bash", "-c", "source activate drGAT && bash" ]
