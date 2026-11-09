FROM nvidia/cuda:13.5.0-devel-ubuntu22.04                                                                                                                                                     
                                                                                                                                                                                              
ENV DEBIAN_FRONTEND=noninteractive                                                                                                                                                            
RUN apt-get update && apt-get upgrade -y && apt-get install -y \                                                                                                                              
      cmake \                                                                                                                                                                                 
      gcc \                                                                                                                                                                                   
      gfortran \
      git \
      build-essential \
      openmpi-bin \
      libopenmpi-dev \
      wget \
      && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PREFIX=/scratch
ENV ARCHIVE_DIR=${PREFIX}/archive
ENV SOURCE_DIR=${PREFIX}/source
ENV BUILD_DIR=${PREFIX}/build
ENV INSTALL_DIR=/opt

RUN mkdir -p ${PREFIX} && \
    cd ${PREFIX} && \
    mkdir archive && \
    mkdir source && \
    mkdir build

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Install Kokkos
ARG KOKKOS_VERSION=5.2.0
RUN export KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    export KOKKOS_ARCHIVE=${ARCHIVE_DIR}/kokkos.tar.bz2 && \
    export KOKKOS_SOURCE_DIR=${SOURCE_DIR}/kokkos && \
    export KOKKOS_BUILD_DIR=${BUILD_DIR}/kokkos && \
    export KOKKOS_INSTALL_DIR=${INSTALL_DIR}/kokkos && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p ${KOKKOS_SOURCE_DIR} && \
    tar -xf ${KOKKOS_ARCHIVE} -C ${KOKKOS_SOURCE_DIR} --strip-components=1 && \
    mkdir -p ${KOKKOS_BUILD_DIR} && \
    cd ${KOKKOS_BUILD_DIR} && \
    cmake -DCMAKE_CXX_COMPILER=${KOKKOS_SOURCE_DIR}/bin/nvcc_wrapper \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DCMAKE_INSTALL_PREFIX=${KOKKOS_INSTALL_DIR} \
        -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
        ${KOKKOS_SOURCE_DIR} && \
    make -j4 install && \
    rm -rf ${KOKKOS_ARCHIVE} && \
    rm -rf ${KOKKOS_BUILD_DIR} && \
    rm -rf ${KOKKOS_SOURCE_DIR}
ENV OMPI_CXX=${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper
