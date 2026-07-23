FROM ros:jazzy-ros-base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
        build-essential python3-dev python3-venv libc-dev \
        ros-jazzy-rmw-zenoh-cpp && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y gnupg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV UV_HTTP_TIMEOUT=300

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY roboclaw/embodied/engine/ ./roboclaw/embodied/engine/

RUN mkdir -p bridge && \
    touch roboclaw/__init__.py && \
    uv pip install --no-cache . && \
    rm -rf roboclaw/__init__.py bridge

COPY roboclaw/ roboclaw/
COPY bridge/ bridge/
RUN uv pip install --no-cache .

WORKDIR /app/bridge
RUN npm install && npm run build
WORKDIR /app

RUN mkdir -p /root/.roboclaw

EXPOSE 18790

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && roboclaw \"$@\"", "--"]
CMD ["status"]