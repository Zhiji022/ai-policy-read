FROM python:3.11
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN mkdir -p ~/my_tempfile && chmod 777 ~/my_tempfile
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["chainlit", "run", "app.py", "--port", "7860"]