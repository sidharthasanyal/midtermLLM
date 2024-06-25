FROM python:3.12.3
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
RUN chmod 777 $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN chmod 777 $HOME/app
CMD ["chainlit", "run", "airbnb.py", "--port", "7660"]