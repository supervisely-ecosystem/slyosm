FROM supervisely/base-py-sdk-light:6.73.564
LABEL supervisely-sdk-version="6.73.564"

WORKDIR /app

COPY dev_requirements.txt /app/dev_requirements.txt
RUN pip install --no-cache-dir -r /app/dev_requirements.txt

COPY . /app
