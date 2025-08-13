## 说明文档

修改分布在两个文件：`hnsw_grap.h` 和 `ghnsw_example_filter.cpp`

### `hnsw_graph.h`

- `addPointLimit()`，两个重载函数用于构建索引
- `searchBaseLayerLimit()`，返回给定点的ANNS，目前仅用于构建阶段
- `getEntryPointForKHop()`，供`searchBaseLayerLimit()`调用，用于选取一个起始点进行扩展，仅用于构建阶段
- `searchKnnLimit()`，返回给定节点在索引上的邻居节点



### `ghnsw_example_filter.cpp`

测试文件，所有可调整参数目前在`main()`开始处定义，修改后需要重新编译该测试文件。

总共测试了四种方法：hnsw + filter，hnsw，暴力算法 + filter和单层图索引（我们的方法）。以暴力算法+filter的结果作为标准计算正确率。
