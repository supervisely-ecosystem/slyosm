FROM supervisely/base-py-sdk:6.73.418
LABEL supervisely-sdk-version="6.73.418"

WORKDIR /app

COPY dev_requirements.txt /app/dev_requirements.txt
RUN pip install --no-cache-dir -r /app/dev_requirements.txt

COPY . /app
