FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get upgrade
RUN apt-get dist-upgrade
RUN apt-get update && apt-get -qq -y install curl		\
&& apt-get install -y -qq automake	                                \
                        autoconf                                \
                        apt-utils                               \
                        vim                                     \
                        git                                     \
                        curl                                    \
                        wget                                    \
                        python3                                 \
                        python3-dev                             \
                        python3-pip                             \
                        python3-sklearn                         \
                        python3-sklearn-lib                     
                        

RUN apt-get install -y -qq lsb-release                              \
                        sudo                                        \
                        systemctl                                   \
                        cmake                                       \
                        bison                                       \
                        mercurial                                    \
                        libpcre2-dev                                \
                        pkg-config                                  \
                        libreadline-dev                             \
                        liblzma-dev                                 \
                        libtool                                     \
                        locales                                     \
                        gettext                                     \
                        r-base		                    			\
                        unixodbc-dev		                		\
                        uuid-dev			                    	\
                        zlib1g-dev			                    	\
                        libcfitsio-dev		                		\
                        liblz4-dev		                    		\
                        libsnappy-dev		                		  \
                        flex                                  \
                        openjdk-21-jre # for benchbase


# install python libraries. Note that AIDA only supports older versions of numpy and scipy.
RUN pip3 install tblib                                           \
  && pip3 install geopy                                          \
  && pip3 install numpy==1.26.4                                  \
  && pip3 install scipy==1.8.0                                   \
  && pip3 install pandas                                         \
  && pip3 install dill                                           \
  && pip3 install six                                            \
  && pip3 install dash                                           \
  && pip3 install py-lz4framed                                   \
  && pip3 install image                                          \
  && pip3 install matplotlib                                     \
  && pip3 install psycopg2-binary                                \
  && pip3 install psycopg                                        \
  && pip3 install pillow                                         \
  && pip3 install py-lz4framed                                   \
  && pip3 install plotly                                         \
  && pip3 install dash dash-core-components dash-html-components flask-cors \
  && pip3 install pickleshare              

RUN pip3 install torch torchvision torchaudio        \
  && pip3 install tensorflow && pip3 install cupy-cuda12x

# Set system locale (used by DB)
RUN locale-gen en_US.UTF-8   

# Build Postgres 17 from source
RUN git clone -b REL_17_STABLE https://github.com/postgres/postgres.git /home/build/postgres ### Takes some time
WORKDIR /home/build/postgres
RUN mkdir -p /home/build/postgres/installdir
RUN /home/build/postgres/configure --prefix=$PWD/installdir   \
    --exec-prefix=$PWD/installdir --with-includes=/usr/local/lib/python3.10/dist-packages/numpy/core/include  \
    --with-python                    \
    &&  make  \
    &&  make install

RUN mkdir -p /home/build/pg_storeddata
COPY *.conf /home/build/postgres/

ENV PATH=/home/build/postgres/installdir/bin:$PATH

# Build Multicorn 2 from source
RUN git clone https://github.com/pgsql-io/multicorn2.git /home/build/multicorn2
WORKDIR /home/build/multicorn2
RUN make && make install

COPY AIDA /home/build/AIDA
COPY AIDA-Benchmarks /home/build/AIDA-Benchmarks

# Install benchbase
RUN git clone --depth 1 https://github.com/cmu-db/benchbase.git /home/build/benchbase
WORKDIR /home/build/benchbase
RUN ./mvnw clean package -P postgres
RUN tar xvzf target/benchbase-postgres.tgz

# Copy scripts to container
RUN chmod +x scripts/**/*
COPY scripts/benchbase/* /home/build/benchbase/
COPY scripts/postgres/* /home/build/postgres/
RUN ln -s /home/build/postgres/env.sh /home/build/AIDA/aidaPostgreSQL/scripts/env.sh # Symbolic link for the env script
# The * makes it optional, since postgres-data.zip is not in the repo by default
COPY postgres-data.zip* /home/build/postgres/data.zip

# Setup user
RUN adduser --quiet --disabled-password --gecos ""  aida-user && chown -R aida-user /home/build \
    && echo "aida-user:aida" | chpasswd && usermod -aG sudo aida-user 
RUN chown aida-user -R /home/build
USER aida-user


WORKDIR /home/build
