#include "../../hnswlib/hnswlib.h"

// Filter that allows labels divisible by divisor
class PickDivisibleIds: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(hnswlib::labeltype label_id) {
        return label_id % divisor == 0;
    }
};

void verifyConnectionConstraints(hnswlib::GraphHNSW<float>* index, hnswlib::GraphRelationSampler* grs, int k) {
    size_t total_connections = 0;
    size_t valid_connections = 0;
    size_t violations = 0;
    
    // 遍历所有点
    for (size_t i = 0; i < index->getCurrentElementCount(); i++) {
        hnswlib::labeltype label_i = index->getExternalLabel(i);
        // 获取该点的k-hop邻居集合
        std::unordered_set<hnswlib::labeltype> khop = grs->getKHopNodes(label_i, k);
        
        // 检查该点的所有层级的所有连接
        for (int level = 0; level <= index->element_levels_[i]; level++) {
            std::vector<hnswlib::tableint> connections = index->getConnectionsWithLock(i, level);
            total_connections += connections.size();
            
            // 检查每个连接是否在k-hop邻居内
            for (hnswlib::tableint conn : connections) {
                hnswlib::labeltype label_conn = index->getExternalLabel(conn);
                if (khop.count(label_conn) || label_conn == label_i) {
                    valid_connections++;
                } else {
                    violations++;
                    std::cout << "违反约束: 点" << label_i << "与非k-hop邻居" 
                              << label_conn << "在第" << level << "层有连接" << std::endl;
                }
            }
        }
    }
    
    float compliance_rate = (float)valid_connections / total_connections * 100;
    std::cout << "k-hop连接约束分析:" << std::endl;
    std::cout << "总连接数: " << total_connections << std::endl;
    std::cout << "有效连接数: " << valid_connections << std::endl;
    std::cout << "违反约束数: " << violations << std::endl;
    std::cout << "约束遵循率: " << compliance_rate << "%" << std::endl;
}


void analyzeLayerCompliance(hnswlib::GraphHNSW<float>* index, hnswlib::GraphRelationSampler* grs, int k) {
    std::vector<size_t> total_per_layer(index->maxlevel_ + 1, 0);
    std::vector<size_t> valid_per_layer(index->maxlevel_ + 1, 0);
    
    for (size_t i = 0; i < index->getCurrentElementCount(); i++) {
        hnswlib::labeltype label_i = index->getExternalLabel(i);
        std::unordered_set<hnswlib::labeltype> khop = grs->getKHopNodes(label_i, k);
        
        for (int level = 0; level <= index->element_levels_[i]; level++) {
            std::vector<hnswlib::tableint> connections = index->getConnectionsWithLock(i, level);
            total_per_layer[level] += connections.size();
            
            for (hnswlib::tableint conn : connections) {
                hnswlib::labeltype label_conn = index->getExternalLabel(conn);
                if (khop.count(label_conn) || label_conn == label_i) {
                    valid_per_layer[level]++;
                }
            }
        }
    }
    
    std::cout << "各层连接约束遵循情况:\n";
    for (int level = 0; level <= index->maxlevel_; level++) {
        if (total_per_layer[level] > 0) {
            float compliance = (float)valid_per_layer[level] / total_per_layer[level] * 100;
            std::cout << "第" << level << "层: " 
                      << valid_per_layer[level] << "/" << total_per_layer[level] 
                      << " (" << compliance << "%)" << std::endl;
        }
    }
}


void compareSearchPerformance(hnswlib::GraphHNSW<float>* limited_index, hnswlib::HierarchicalNSW<float>* original_index, 
                                float* queries, int num_queries, int dim, int k) {
    double limited_time = 0, original_time = 0;
    double limited_recall = 0;
    
    for (int i = 0; i < num_queries; i++) {
        // 限制版搜索
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result_limited = limited_index->searchKnn(queries + i * dim, k);
        auto t2 = std::chrono::high_resolution_clock::now();
        limited_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        
        // 原版搜索（作为基准）
        t1 = std::chrono::high_resolution_clock::now();
        auto result_original = original_index->searchKnn(queries + i * dim, k);
        t2 = std::chrono::high_resolution_clock::now();
        original_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        
        // 计算召回率（限制版与原版的结果对比）
        std::unordered_set<hnswlib::labeltype> original_labels;
        while (!result_original.empty()) {
            original_labels.insert(result_original.top().second);
            result_original.pop();
        }
        
        int matches = 0;
        while (!result_limited.empty()) {
            if (original_labels.count(result_limited.top().second))
                matches++;
            result_limited.pop();
        }
        
        limited_recall += (double)matches / k;
    }
    
    limited_time /= num_queries;
    original_time /= num_queries;
    limited_recall /= num_queries;
    
    std::cout << "搜索性能对比:\n";
    std::cout << "限制版平均搜索时间: " << limited_time << "微秒\n";
    std::cout << "原版平均搜索时间: " << original_time << "微秒\n";
    std::cout << "限制版与原版的平均召回率: " << limited_recall * 100 << "%\n";
}


int main() {
    int dim = 16;               // 维度
    int max_elements = 10000;   // 最大元素数
    int M = 16;                 // HNSW参数
    int ef_construction = 200;  // 构建参数
    int k_query = 10;           // 查询时返回的邻居数
    int num_queries = 100;      // 测试查询次数

    int k_hop = 2; // k-hop参数
    float prob = 0.01; // 建边概率，不宜太低，否则容易出现 SIGSEGV 错误

    // 初始化空间
    hnswlib::L2Space space(dim);
    
    // 创建两个索引实例
    hnswlib::GraphHNSW<float>* limited_index = new hnswlib::GraphHNSW<float>(&space, max_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* original_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // 初始化图关系采样器(仅限制版本使用)
    hnswlib::GraphRelationSampler grs = hnswlib::GraphRelationSampler(prob);
    limited_index->setGraphHop(&grs, k_hop);

    // 生成随机数据
    std::mt19937 rng(47);
    std::uniform_real_distribution<> distrib;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib(rng);
    }

    // 将数据添加到两个索引中
    for (int i = 0; i < max_elements; i++) {
        limited_index->addPointLimit(data + i * dim, i);
        original_index->addPoint(data + i * dim, i);
    }

    // 生成查询数据
    float* queries = new float[dim * num_queries];
    for (int i = 0; i < dim * num_queries; i++) {
        queries[i] = distrib(rng);
    }

    // 设置搜索参数
    limited_index->setEf(50);
    original_index->setEf(50);


    // 验证连接约束
    std::cout << "验证连接约束..." << std::endl;
    verifyConnectionConstraints(limited_index, &grs, k_hop);
    analyzeLayerCompliance(limited_index, &grs, k_hop);
    std::cout << "连接约束验证完成" << std::endl;

    // 比较搜索性能
    std::cout << "\n开始比较搜索性能..." << std::endl;
    compareSearchPerformance(limited_index, original_index, queries, num_queries, dim, k_query);
    std::cout << "搜索性能比较完成" << std::endl;
    
    // 清理资源
    delete[] data;
    delete[] queries;
    delete limited_index;
    delete original_index;
    
    return 0;
}
