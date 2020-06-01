FROM nvidia/cuda:10.0-base-ubuntu18.04


# install Python
ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip \
    wget

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
	
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

	


RUN mkdir -p /work

WORKDIR /work

COPY requirements.txt .

RUN pip install -r requirements.txt \
    && pip install jupyter_http_over_ws \
    && jupyter serverextension enable --py jupyter_http_over_ws
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# Installing Node
RUN apt-get update \
        && apt-get -yq install curl gnupg wget \
        && curl -sL https://deb.nodesource.com/setup_12.x  | bash - \
        && apt-get -yq install nodejs
# Install jupyterlab
RUN pip install jupyterlab \
        && pip install jupyter-tensorboard \
        && pip install jupyter-lsp \
        && pip install ipywidgets \
        && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
        && jupyter nbextension enable --py widgetsnbextension \
        && jupyter labextension install @jupyterlab/toc \
        && jupyter labextension install @ijmbarr/jupyterlab_spellchecker \
        && jupyter labextension install jupyterlab_tensorboard \
        && jupyter labextension install @lckr/jupyterlab_variableinspector \
        && jupyter labextension install @aquirdturtle/collapsible_headings \
        && jupyter labextension install @mohirio/jupyterlab-horizon-theme \
        && jupyter labextension install @krassowski/jupyterlab-lsp  \
        && pip install python-language-server[all] \
        && jupyter labextension install @telamonian/theme-darcula \
        && jupyter lab build

RUN jupyter serverextension enable --sys-prefix --py jupyter_lsp \
    && jupyter labextension install @wallneradam/output_auto_scroll \
        && jupyter lab build



EXPOSE 8081

CMD ["jupyter","lab","--allow-root","--port","8081","--NotebookApp.port_retries=0","--ip","0.0.0.0"]
