FROM mdstudio/mdstudio_docker_conda:0.0.3

# Set permissions and install package
COPY . /home/mdstudio
RUN chown -R mdstudio:mdstudio /home/mdstudio
RUN chmod -R 755 /home/mdstudio
WORKDIR /home/mdstudio

RUN conda install numpy scipy subprocess32
RUN pip install .

USER mdstudio

CMD ["bash", "entry_point_mdstudio_pylie.sh"]
