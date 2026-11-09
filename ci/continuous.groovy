pipeline {
  agent none

    options {
      disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 6, unit: 'HOURS')
    }

  triggers {
    issueCommentTrigger('.*test this please.*')
  }

  stages {
    stage('Test'){
      parallel {
        stage('Kokkos 5.2.0') {
          agent {
            dockerfile {
              filename 'Dockerfile'
                dir 'ci'
                label 'nvidia-docker && ampere'
                args '--env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
            }
          }
          steps {
            sh '''#!/bin/bash
              rm -rf build && mkdir -p build && cd build && \
              cmake \
              -DYAKL_USE_INSTALLED_KOKKOS=ON \
              -DCMAKE_CXX_COMPILER=/opt/kokkos/bin/nvcc_wrapper \
              -DCMAKE_CXX_EXTENSIONS=OFF \
              ../unit
              make -j8
              ctest -V --no-compress-output -T Test
            '''
          }
          post {
            always {
              xunit([CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)])
            }
          }
        }
      }
    }
  }
}
