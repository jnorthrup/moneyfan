FROM maven:3.9-eclipse-temurin-21

# Install essential tools
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install SDKMAN
RUN curl -s "https://get.sdkman.io" | bash
ENV SDKMAN_DIR="/root/.sdkman"
RUN bash -c "source $SDKMAN_DIR/bin/sdkman-init.sh && \
    sdk install java latest && \
    sdk install maven latest && \
    sdk install kotlin latest"

# Set environment variables
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=${JAVA_HOME}/bin:${PATH}
ENV PATH=${SDKMAN_DIR}/candidates/java/current/bin:${PATH}
ENV PATH=${SDKMAN_DIR}/candidates/maven/current/bin:${PATH}
ENV PATH=${SDKMAN_DIR}/candidates/kotlin/current/bin:${PATH}

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd -m -s /bin/bash developer
RUN chown -R developer:developer /app
USER developer

# Default command
CMD ["bash"]