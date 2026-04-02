FROM buildpack-deps:bookworm

ENV ELAN_HOME=/usr/local/elan \
    PATH=/usr/local/elan/bin:$PATH \
    LEAN_VERSION=leanprover/lean4:nightly

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain $LEAN_VERSION && \
    elan default $LEAN_VERSION && \
    elan --version && \
    lean --version && \
    leanc --version && \
    lake --version

CMD ["sleep", "infinity"]
