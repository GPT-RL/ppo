FROM nvcr.io/nvidia/pytorch:21.07-py3 AS base

# Setup locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# no .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# traceback on segfau8t
ENV PYTHONFAULTHANDLER 1

# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

COPY env.yml .
RUN conda env update -n base -f env.yml \
 && pip install git+https://github.com/ethanabrooks/sweep-logger.git@087a2a7869224203a3dba40f800571faa2c0f117
WORKDIR "/project"
COPY . .
ENTRYPOINT ["python"]
