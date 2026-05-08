# Official Code for Paper "Approximate Nearest Neighbor Search with Graph Range Filters"



## Introduction



This is the official code for the paper Approximate Nearest Neighbor Search with Graph Range Filters.



The code contains the implementation of the proposed algorithms DAL, DLH and DLH-M, and the compared baselines Pre-Filter, HNSW, ACORN and Navix.



## Run Code



Please following the following steps for data preparation and code compilation.



#### Data Preparation

```shell
python main_gen.py
```

#### Code Compilation

```shell
cd build
cmake ..
make
```



#### Code Execution

```shell
python main_test.py
```


## Acknowledgement



DLH is built based on [hnswlib](https://github.com/nmslib/hnswlib) and [C++ Bloom filter library](https://github.com/ArashPartow/bloom).

