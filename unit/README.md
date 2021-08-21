# Running the YAKL unit tests

```bash
git clone git@github.com:mrnorman/YAKL.git
git submodule update --init
cd unit/build
cd machines/[machine_name]
source machine_option.sh
make -j8
make test
```

