# Version

## pslV3.h
hnsw + psl (use query_by_t)

## pslV5.h
hnsw + psl + bloom filter

## 

# DLH



## 图关系以及groundtruth生成

data/gen_groundtruth.cpp: 根据给定参数生成图关系和groundtruth

main_gen.py: 主要的生成脚本，设置参数传入并执行data/gen_groundtruth.cpp



## DLH实现

hnswlib/bloomfilter.h: 布隆过滤器

hnswlib/pslV3.h: DAL

hnswlib/pslV5.h: DLH

hnswlib/pslV7.h: DLH-M



## 实验

data/baseline: 各个方法的实验代码

data/common: 部分共用逻辑



## 日志与绘图

logs: 实验日志

main_page_gen.py: 主图绘制脚本

bar_plot_gen.py: index_size实验图绘制脚本

param_fpp_qps_recall.py: FPP实验图绘制脚本

param_khop_qps_recall.py: khop变化实验绘图脚本

scala_qps_recall.py: graph range变化实验绘图脚本

figures: 实验结果图
