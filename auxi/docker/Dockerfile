FROM python:3.11-bullseye

WORKDIR /opt/app

RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip

# Disable automatic cache cleanup so we can persist the apt cache between build runs
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# RUN --mount=type=cache,target=/var/cache/apt  --mount=type=cache,target=/var/lib/apt \
# apt-get update && export DEBIAN_FRONTEND=noninteractive \
# && apt-get -y install --no-install-recommends \

# COPY docker/requirements-dev.txt /tmp/requirements.txt

# RUN --mount=type=cache,target=/root/.cache \
#     pip3 install -r /tmp/requirements.txt

COPY . .

# RUN --mount=type=cache,target=/root/.cache \
#     pip3 install --no-deps .

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -e '.[dev]'

EXPOSE 80

CMD ["sh", "/opt/app/docker/entrypoint.sh"]
