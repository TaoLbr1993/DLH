#include "../../hnswlib/hnswlib.h"
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

class PSLFilter : public hnswlib::BaseFilterFunctor {
    public:
    int src_id;
    int qhop;
    hnswlib::DisOracle * pslidx;
    PSLFilter(int src_id_, int qhop_, hnswlib::DisOracle* psl_index_) : src_id(src_id_), qhop(qhop_), pslidx(psl_index_) {}

    bool operator() (hnswlib::labeltype id) override {
        return pslidx->query(src_id, id, qhop);
    }
};

// 定义一个k跳邻居过滤器
class KHopFilter : public hnswlib::BaseFilterFunctor {
public:
    KHopFilter(const std::unordered_set<hnswlib::labeltype>& nbrs) : khop_neighbors(nbrs) {}
    
    bool operator()(hnswlib::labeltype id) override {
        return khop_neighbors.count(id) > 0;
    }
    
private:
    const std::unordered_set<hnswlib::labeltype>& khop_neighbors;
};

// 封装近似搜索函数
std::priority_queue<std::pair<float, hnswlib::labeltype>>
searchKnnFilterV1(const void* query_data, size_t k, hnswlib::labeltype query_label, 
               hnswlib::HierarchicalNSW<float>* index, hnswlib::GraphRelationSampler* grs, int k_hop) {
    // 获取查询点的k-hop邻居集合
    std::unordered_set<hnswlib::labeltype> khop_nbr = grs->getKHopNodes(query_label, k_hop);
    
    // 创建过滤器
    KHopFilter filter(khop_nbr);
    
    // 使用过滤器进行搜索
    return index->searchKnn(query_data, k, &filter);
}

std::priority_queue<std::pair<float, hnswlib::labeltype>> 
searchKnnFilter(const void* query_data, size_t k, hnswlib::labeltype query_label, 
               hnswlib::HierarchicalNSW<float>* index, hnswlib::GraphRelationSampler* grs, int k_hop) {
    // 获取查询点的k-hop邻居集合
    std::unordered_set<hnswlib::labeltype> khop_nbr = grs->getKHopNodes(query_label, k_hop);
    
    // 创建过滤器
    KHopFilter filter(khop_nbr);
    
    // 使用过滤器进行搜索
    return index->searchKnn(query_data, k, &filter);
               }

// 封装精确搜索函数
std::priority_queue<std::pair<float, hnswlib::labeltype>>
exactKnnWithGraphLimit(const void* query_data, size_t k, hnswlib::labeltype query_label,
                      hnswlib::BruteforceSearch<float>* bf_index, hnswlib::GraphRelationSampler* grs, int k_hop) {
    // 获取查询点的k-hop邻居集合
    std::unordered_set<hnswlib::labeltype> khop_nbr = grs->getKHopNodes(query_label, k_hop);
    
    // 创建过滤器
    KHopFilter filter(khop_nbr);
    
    // 使用过滤器进行精确搜索
    return bf_index->searchKnn(query_data, k, &filter);
}

// 封装使用GraphHNSW直接搜索的函数
std::priority_queue<std::pair<float, hnswlib::labeltype>>
searchKnnLimit(size_t k, hnswlib::labeltype query_label, 
                     hnswlib::GraphHNSW<float>* index) {
    return index->searchKnnLimit(k, query_label);
}

// graphHNSW改造为可以接受多个query，因此我们在这个封装函数里就把结果全部获取并返回
 std::priority_queue<std::pair<float, hnswlib::labeltype>>
searchKnnLimitMultiQuery(size_t k, hnswlib::labeltype query_label, 
                     std::vector<hnswlib::GraphHNSW<float>*> ghnsw_list,
                     std::vector<int> hop_partitions) {
    // 记录已处理的节点
    std::unordered_set<hnswlib::labeltype> processed_nodes;
    // 创建一个存储每层结果的数组，all_results[i] 表示第 i 次的结果
    std::unordered_set<hnswlib::labeltype> all_results;

    // 记录最近的k个结果
    std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates;
    // 创建一个仅包含查询点的优先队列作为初始结果(第 0 次)
    all_results.insert(query_label);
    
    for (int query_cnt=0; query_cnt<hop_partitions.size(); query_cnt++) {
        
        std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> current_layer_results;
        ghnsw_list[hop_partitions[query_cnt]]->searchKnnLimitMultiquery(k, query_label, all_results, candidates);
    }
    return candidates;
}

// 验证结果是否在k-hop范围内
void verifyResults(const std::priority_queue<std::pair<float, hnswlib::labeltype>>& results,
                 const std::unordered_set<hnswlib::labeltype>& khop_nbr, const hnswlib::labeltype query_label,
                 const std::string& source) {
    auto results_copy = results;
    while (!results_copy.empty()) {
        auto label = results_copy.top().second;
        if (label != query_label && khop_nbr.count(label) == 0) {
            std::cerr << "Error: Result " << label << " from " << source 
                      << " is not in k-hop neighbors of query label " << query_label << std::endl;
            assert(false);
        }
        assert(khop_nbr.count(label) > 0 || label == query_label);
        results_copy.pop();
    }
}


// 工具函数：合并多个优先队列并去重
std::priority_queue<std::pair<float, hnswlib::labeltype>> 
mergeQueuesUnique(const std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>>& queues, const hnswlib::labeltype query_label) {
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    std::unordered_set<hnswlib::labeltype> seen_labels;

    seen_labels.insert(query_label); // 确保查询点本身不会重复添加
    
    // 直接遍历所有队列，将不重复的元素加入结果
    for (const auto& queue : queues) {
        auto queue_copy = queue;
        while (!queue_copy.empty()) {
            auto element = queue_copy.top();
            if (seen_labels.count(element.second) == 0) {
                // 添加到结果集中
                result.push(element);
                seen_labels.insert(element.second);
            }
            queue_copy.pop();
        }
    }
    
    return result;
}

// 统计k跳邻居占总节点比例的函数
void analyzeKHopDistribution(hnswlib::GraphRelationSampler& grs, int max_elements, int max_hops = 5) {
    std::cout << "\n================ k跳邻居分布统计 =================" << std::endl;
    std::cout << "Hop距离    平均邻居数量    占总节点比例(%)" << std::endl;
    
    // 随机选择样本节点进行统计
    const int num_samples = 100;  // 样本数量
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, max_elements - 1);
    
    // 对每个hop层级进行统计
    for (int k = 1; k <= max_hops; k++) {
        long total_neighbors = 0;
        
        for (int i = 0; i < num_samples; i++) {
            int node_id = uni(rng);
            auto khop_nbrs = grs.getKHopNodes(node_id, k);
            total_neighbors += khop_nbrs.size();
        }
        
        double avg_neighbors = static_cast<double>(total_neighbors) / num_samples;
        double percentage = (avg_neighbors / max_elements) * 100.0;
        
        std::cout << std::setw(5) << k << "跳"
                  << std::setw(16) << avg_neighbors
                  << std::setw(20) << std::fixed << std::setprecision(2) << percentage << std::endl;
    }
}

int main() {
    int dim = 128;               // 维度
    int max_elements = 50000;   // 最大元素数
    int M = 80;                 // 最大连接数

    int nbrM = 4;

    int ef_construction = 200;  
    int k_query = 50;           // 查询时返回的邻居数
    int num_queries = 100;      // 测试查询次数

    int k_hop = 4 ;              // k-hop参数
    float prob = 0.0003;          // 建边概率

    // 定义跳数划分，各部分之和等于k_hop
    // std::vector<int> hop_partitions = {1,1,1,1};

    // // 确保划分之和等于k_hop
    // int sum_hops = 0;
    // for (int hop : hop_partitions) sum_hops += hop;
    // assert(sum_hops == k_hop && "跳数划分之和必须等于k_hop");

    // 初始化空间
    hnswlib::L2Space space(dim);

    // 初始化图关系采样器
    hnswlib::GraphRelationSampler grs(prob);
    size_t * ids = new size_t[max_elements];
    for (size_t i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    
    grs.genRelation(ids, max_elements);
    grs.printInfo();

    // 生成随机数据
    std::mt19937 rng(47);
    std::uniform_real_distribution<> distrib;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib(rng);
    }

    // 将数据添加到HNSW索引中
    std::cout << "正在构建HNSW索引..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float>* normal_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    for (int i = 0; i < max_elements; i++) {
        normal_index->addPoint(data + i * dim, i);
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    auto hnsw_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "HNSW索引构建完成，耗时: " << hnsw_build_time << " 毫秒" << std::endl;

    // 创建暴力搜索索引（用于精确查询）
    std::cout << "正在构建暴力搜索索引..." << std::endl;
    build_start = std::chrono::high_resolution_clock::now();
    hnswlib::BruteforceSearch<float>* bf_index = new hnswlib::BruteforceSearch<float>(&space, max_elements);
    for (int i = 0; i < max_elements; i++) {
        bf_index->addPoint(data + i * dim, i);
    }
    build_end = std::chrono::high_resolution_clock::now();
    auto bf_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "暴力搜索索引构建完成，耗时: " << bf_build_time << " 毫秒" << std::endl;

    // 创建HNSW+PSL
    std::cout << "正在构建HNSW+PSL..." << std::endl;
    build_start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float>* hnsw_psl_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    for (int i = 0; i < max_elements; i++) {
        hnsw_psl_index->addPoint(data + i * dim, i);
    }
    hnswlib::DisOracle psl_index(grs.edge_pairs, k_hop, false);
    psl_index.see_labels();

    // psl_index.create_ord(grs.edge_pairs);
    // TODO: construct

    build_end = std::chrono::high_resolution_clock::now();
    auto hnswpsl_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "HNSW+PSL构建完成，耗时: " << hnswpsl_build_time << " 毫秒" << std::endl;
    std::cout << "HNSW+PSL: HNSW index size: " << hnsw_psl_index->indexFileSize() << std::endl;


    // 召回率
    double total_recall_with_filter = 0.0;
    double total_recall_no_filter = 0.0;
    double total_recall_ghnsw = 0.0;
    double total_recall_hnswgdist = 0.0;
    double total_recall_hnswpsl = 0.0;
    
    // 查询时间
    double total_time_hnsw_filter = 0.0;      // HNSW带过滤器的总查询时间
    double total_time_hnsw_no_filter = 0.0;   // HNSW不带过滤器的总查询时间
    double total_time_bf = 0.0;               // 暴力搜索的总查询时间
    double total_time_ghnsw = 0.0;      // 单层图索引的总查询时间
    double total_time_hnswgdist = 0.0;
    double total_time_hnswpsl = 0.0;

    // 随机选择查询点进行测试
    std::cout << "\n开始评估性能..." << std::endl;
    for (int i = 0; i < num_queries; i++) {
        // 随机选择一个已有点作为查询点
        int query_idx = rng() % max_elements;
        float* query_vector = data + query_idx * dim;
        hnswlib::labeltype query_label = query_idx;

        // 获取查询点的k-hop邻居集合（用于后续验证）
        std::unordered_set<hnswlib::labeltype> khop_nbr = grs.getKHopNodes(query_label, k_hop);

        // ==================== 带过滤器的HNSW ====================
        auto t1 = std::chrono::high_resolution_clock::now();
        // 获取查询点的k-hop邻居集合
        std::unordered_set<hnswlib::labeltype> khop_nbr_hnswbf = grs.getKHopNodes(query_label, k_hop);
        
        // 创建过滤器
        KHopFilter hnswbf_filter(khop_nbr_hnswbf);
        
        // 使用过滤器进行搜索
        auto approximate_results_filtered = normal_index->searchKnn(query_vector, k_query, &hnswbf_filter);
        //  = searchKnnFilter(query_vector, k_query, query_label, normal_index, &grs, k_hop);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto approx_time_filtered = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        // 验证结果
        verifyResults(approximate_results_filtered, khop_nbr, query_label, "带过滤器HNSW");
        
        // ==================== 带过滤器的BF ====================
        t1 = std::chrono::high_resolution_clock::now();
        auto exact_results = exactKnnWithGraphLimit(query_vector, k_query, query_label, bf_index, &grs, k_hop);
        t2 = std::chrono::high_resolution_clock::now();
        auto exact_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        
        // 验证结果
        verifyResults(exact_results, khop_nbr, query_label, "暴力搜索");
        
        // 计算用于召回率的精确结果集合
        std::unordered_set<hnswlib::labeltype> exact_labels;
        auto exact_results_copy = exact_results;
        while (!exact_results_copy.empty()) {
            exact_labels.insert(exact_results_copy.top().second);
            exact_results_copy.pop();
        }
        
        total_time_bf += exact_time;

        // 计算带过滤器HNSW的召回率
        int matches_filtered = 0;
        auto approx_results_filtered_copy = approximate_results_filtered;
        while (!approx_results_filtered_copy.empty()) {
            if (exact_labels.count(approx_results_filtered_copy.top().second) > 0) {
                matches_filtered++;
            }
            approx_results_filtered_copy.pop();
        }
        
        double recall_filtered = (double)matches_filtered / k_query;
        total_recall_with_filter += recall_filtered;
        total_time_hnsw_filter += approx_time_filtered;

        // ==================== 不带过滤器的HNSW测试 ====================
        t1 = std::chrono::high_resolution_clock::now();
        auto approximate_results_unfiltered = normal_index->searchKnn(query_vector, k_query);
        auto t2_unfiltered = std::chrono::high_resolution_clock::now();
        auto approx_time_unfiltered = std::chrono::duration_cast<std::chrono::microseconds>(t2_unfiltered - t1).count();
                
        // 计算不带过滤器HNSW的召回率
        int matches_unfiltered = 0;
        auto approx_results_unfiltered_copy = approximate_results_unfiltered;
        while (!approx_results_unfiltered_copy.empty()) {
            if (exact_labels.count(approx_results_unfiltered_copy.top().second) > 0) {
                matches_unfiltered++;
            }
            approx_results_unfiltered_copy.pop();
        }

        double recall_unfiltered = (double)matches_unfiltered / k_query;
        total_recall_no_filter += recall_unfiltered;
        total_time_hnsw_no_filter += approx_time_unfiltered;

        // 计算HNSW-PSL
        t1 = std::chrono::high_resolution_clock::now();
        psl_index.init_query_node(query_label);
        PSLFilter hnswpsl_filter(query_label, k_hop, &psl_index);

        auto approximate_results_hnswpsl = normal_index->searchKnn(query_vector, k_query, &hnswpsl_filter);
        auto t2_hnswpsl = std::chrono::high_resolution_clock::now();
        auto approx_time_hnswpsl = std::chrono::duration_cast<std::chrono::microseconds>(t2_hnswpsl - t1).count();
        
        verifyResults(exact_results, khop_nbr, query_label, "暴力搜索");

        // 计算HNSW-PLL
        int matches_hnswpsl = 0;
        auto approx_results_hnswpsl_copy = approximate_results_hnswpsl;
        while (!approx_results_hnswpsl_copy.empty()) {
            if (exact_labels.count(approx_results_hnswpsl_copy.top().second) > 0) {
                matches_hnswpsl++;
            }
            approx_results_hnswpsl_copy.pop();
        }
        total_recall_hnswpsl += ((double)matches_hnswpsl / k_query);
        total_time_hnswpsl += approx_time_hnswpsl;

        // 进度显示
        if ((i + 1) % 10 == 0) {
            std::cout << "已完成 " << (i+1) << "/" << num_queries << " 次查询" << std::endl;
        }
    }

    // 与其他方法比较
    std::cout << "\n================ 各方法比较 =================" << std::endl;
    std::cout << "                  构建时间(毫秒)  查询时间(微秒)  召回率(%)" << std::endl;
    std::cout << "暴力搜索+过滤器:    " << std::setw(14) << bf_build_time 
          << std::setw(16) << (total_time_bf / num_queries)
          << std::setw(12) << "100.00" << std::endl;
    std::cout << "HNSW+过滤器:        " << std::setw(14) << hnsw_build_time 
          << std::setw(16) << (total_time_hnsw_filter / num_queries)
          << std::setw(12) << (total_recall_with_filter / num_queries * 100) << std::endl;
    std::cout << "HNSW不带过滤器:     " << std::setw(14) << hnsw_build_time 
          << std::setw(16) << (total_time_hnsw_no_filter / num_queries)
          << std::setw(12) << (total_recall_no_filter / num_queries * 100) << std::endl;

    std::cout << "HNSW-PSL:     " << std::setw(14) << hnswpsl_build_time 
          << std::setw(16) << (total_time_hnswpsl / num_queries)
          << std::setw(12) << (total_recall_hnswpsl / num_queries * 100) << std::endl;

    // 在main函数结束前调用
    analyzeKHopDistribution(grs, max_elements, 5); // 分析1到5跳的邻居分布

    // 清理资源
    delete[] data;
    delete[] ids;
    delete normal_index;
    delete bf_index;
    
    // 只删除非空的索引
    // for (auto index : ghnsw_indices) {
    //     if (index != nullptr) {
    //         delete index;
    //     }
    // }
    
    return 0;
}
