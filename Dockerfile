from python:3.9-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install connexion[swagger-ui]

COPY ./app /usr/src/app

WORKDIR /usr/src/
EXPOSE 1975

ENTRYPOINT [ "gunicorn" ]
CMD ["-w", "4","--access-logfile", "'-'","-b", "0.0.0.0:1975", "app.wsgi"]
