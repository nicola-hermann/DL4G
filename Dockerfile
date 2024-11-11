FROM python:3.11.10-bookworm

ENV PYTHONUNBUFFERED True
ENV APP_HOME /back-end

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create and activate the virtual environment
RUN python -m venv /env
ENV PATH="/env/bin:$PATH"

WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 20 app:app