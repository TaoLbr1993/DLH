#include "../../hnswlib/hnswlib.h"
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>

#include <faiss/IndexHNSW.h>
#include "faiss/index_io.h"


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
    // 数据集参数
    int dim = 128;               // 维度
    int max_elements = 100000;   // 最大元素数
    float prob = 0.0003;          // 建边概率

    // HNSW参数
    int M = 16;                 // 最大连接数
    int ef_construction = 200;
    std::vector<size_t> efs = { 10, 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800 };
    
    // 查询参数
    int k_query = 10;           // 查询时返回的邻居数
    int num_queries = 100;      // 测试查询次数
    int k_hop = 4;              // k-hop参数

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

    // 创建暴力搜索索引（用于精确查询）
    std::cout << "正在构建暴力搜索索引..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();
    hnswlib::BruteforceSearch<float>* bf_index = new hnswlib::BruteforceSearch<float>(&space, max_elements);
    for (int i = 0; i < max_elements; i++) {
        bf_index->addPoint(data + i * dim, i);
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    auto bf_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "暴力搜索索引构建完成，耗时: " << bf_build_time << " 毫秒" << std::endl;


    // 随机选择查询点进行测试
    std::cout << "\n开始评估性能..." << std::endl;

    // 统一抽样查询集合（对每个 ef 复用），并预先计算精确结果
    std::vector<int> query_ids;
    query_ids.reserve(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        query_ids.push_back(rng() % max_elements);
    }

    std::vector<std::unordered_set<hnswlib::labeltype>> exact_label_sets(num_queries);

    for (int qi = 0; qi < num_queries; ++qi) {
        int query_idx = query_ids[qi];
        float* query_vector = data + query_idx * dim;
        hnswlib::labeltype query_label = query_idx;

        // 精确的 Top-k 作为真值（带 k-hop 过滤）
        auto exact_results = exactKnnWithGraphLimit(query_vector, k_query, query_label, bf_index, &grs, k_hop);
        std::unordered_set<hnswlib::labeltype> exact_labels;
        while (!exact_results.empty()) {
            exact_labels.insert(exact_results.top().second);
            exact_results.pop();
        }
        exact_label_sets[qi] = std::move(exact_labels);
    }

    // 生成日志（与 main_page_gen.py 的 parse_search_times 匹配）
    std::string group = "RANDOM-d" + std::to_string(dim) + "-n" + std::to_string(max_elements);
    std::string log_dir = std::string("../logs/") + group;        // logs 已保证存在，这里只创建一层子目录
    if (mkdir(log_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "创建目录失败: " << log_dir << std::endl;
        return 1;
    }
    

    // ===================== Navix 评测 =====================

    // 构建 Faiss HNSW (Navix) 索引
    std::cout << "正在构建Navix(HNSW-Flat)索引..." << std::endl;
    auto navix_build_start = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat navix_index(dim, M, faiss::METRIC_L2);
    navix_index.hnsw.efConstruction = ef_construction;
    navix_index.add(max_elements, data);
    auto navix_build_end = std::chrono::high_resolution_clock::now();
    auto navix_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(navix_build_end - navix_build_start).count();
    std::cout << "Navix(HNSW-Flat)索引构建完成，耗时: " << navix_build_ms << " 毫秒" << std::endl;

    // 评测 Navix（与其他基线相同 ef 列表，k-hop 过滤）
    std::vector<std::pair<double,int>> navix_recall_us_pairs;

    // 为单查询接口准备临时存储
    std::vector<float> navix_dist(k_query);
    std::vector<faiss::idx_t> navix_labels(k_query);
    std::vector<char> navix_filter(max_elements);

    for (size_t ef : efs) {
        navix_index.hnsw.efSearch = static_cast<int>(ef);

        long long navix_sum_us = 0;
        double navix_sum_recall = 0.0;

        for (int qi = 0; qi < num_queries; ++qi) {
            int query_idx = query_ids[qi];
            float* query_vector = data + query_idx * dim;
            hnswlib::labeltype query_label = query_idx;

            auto t1 = std::chrono::high_resolution_clock::now();
            // 构建该查询的过滤掩码（k-hop 内为 1）
            std::fill(navix_filter.begin(), navix_filter.end(), 0);
            {
                auto khop_nbrs = grs.getKHopNodes(query_label, k_hop);
                for (auto id : khop_nbrs) {
                    navix_filter[id] = 1;
                }
            }

            faiss::VisitedTable visited(max_elements);
            faiss::HNSWStats stats;

            
            navix_index.navix_single_search(
                query_vector,
                k_query,
                navix_dist.data(),
                navix_labels.data(),
                navix_filter.data(),
                visited,
                stats
            );
            auto t2 = std::chrono::high_resolution_clock::now();
            auto navix_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            navix_sum_us += navix_time_us;

            // 召回率：与真值 Top-k 交集
            int matches = 0;
            for (int r = 0; r < k_query; ++r) {
                auto lid = navix_labels[r];
                if (lid >= 0 && exact_label_sets[qi].count(static_cast<hnswlib::labeltype>(lid)) > 0) {
                    matches++;
                }
            }
            navix_sum_recall += static_cast<double>(matches) / static_cast<double>(k_query);
        }

        double navix_avg_recall = navix_sum_recall / static_cast<double>(num_queries);
        int navix_avg_us = static_cast<int>(std::llround(static_cast<double>(navix_sum_us) / num_queries));
        navix_recall_us_pairs.emplace_back(navix_avg_recall * 100.0, navix_avg_us);

        std::cout << "[NAVIX] ef=" << ef << "  avg_recall=" << std::fixed << std::setprecision(3)
                  << (navix_avg_recall * 100.0) << "%  avg_time=" << navix_avg_us << " us\n";
    }

    // 写 NAVIX 日志（供 main_page_gen.py 解析）
    std::ostringstream oss_navix;
    oss_navix << "Benchmark Report\n";
    oss_navix << "Search Times (ns):\n";
    oss_navix << "Index \\ ef |";
    for (size_t ef : efs) oss_navix << " ef=" << ef;
    oss_navix << "\n";

    oss_navix << "NAVIX: ";
    oss_navix << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < navix_recall_us_pairs.size(); ++i) {
        oss_navix << "(" << navix_recall_us_pairs[i].first << ", " << navix_recall_us_pairs[i].second << " us)";
        if (i + 1 < navix_recall_us_pairs.size()) oss_navix << " ";
    }
    oss_navix << "\n\n----------------------------------------\n";

    std::ofstream fout_navix((log_dir + "/NAVIX.log").c_str(), std::ios::out | std::ios::trunc);
    fout_navix << oss_navix.str();
    fout_navix.close();
    std::cout << "日志已写入: " << (log_dir + "/NAVIX.log") << std::endl;

    return 0;
}
