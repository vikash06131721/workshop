FROM ubuntu:18.04
FROM python:3.8.5



RUN apt-get update -y
RUN apt-get install python-pip -y

RUN apt-get -y update && apt-get install -y --no-install-recommends apt-utils apt-transport-https

RUN apt-get update




# RUN pip install -r req.txt
WORKDIR /app

# Requirements are installed here to ensure they will be cached.
RUN mkdir -p requirements
COPY production.txt requirements/
RUN pip install --no-cache-dir -r requirements/production.txt \
    && rm -rf /requirements

ADD . /app

EXPOSE 9001

CMD ["sh", "-c", "python3 api_model.py"]
