FROM mdstudio/mdstudio_docker_conda:0.0.3

COPY . /home/mdstudio/lie_pylie

RUN chown mdstudio:mdstudio /home/mdstudio/lie_pylie

WORKDIR /home/mdstudio/lie_pylie

RUN conda install numpy scipy subprocess32

RUN pip install .

CMD ["bash", "entry_point_lie_pylie.sh"]
