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

#include <faiss/IndexACORN.h>


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
    int max_elements = 200000;   // 最大元素数
    float prob = 0.0001;          // 建边概率

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
    

    // ===================== HNSW 评测 =====================

    // 将数据添加到HNSW索引中
    std::cout << "正在构建HNSW索引..." << std::endl;
    build_start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float>* normal_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    for (int i = 0; i < max_elements; i++) {
        normal_index->addPoint(data + i * dim, i);
    }
    build_end = std::chrono::high_resolution_clock::now();
    auto hnsw_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "HNSW索引构建完成，耗时: " << hnsw_build_time << " 毫秒" << std::endl;

    // 为每个 ef 统计 (recall, us)
    std::vector<std::pair<double,int>> recall_us_pairs; // {recall_percent, avg_us}

    for (size_t ef : efs) {
        normal_index->setEf(ef);

        long long sum_us = 0;
        double sum_recall = 0.0;

        for (int qi = 0; qi < num_queries; ++qi) {
            int query_idx = query_ids[qi];
            float* query_vector = data + query_idx * dim;
            hnswlib::labeltype query_label = query_idx;

            // 计时 + 近似检索（带过滤）
            auto t1 = std::chrono::high_resolution_clock::now();
            // 过滤器（k-hop）
            std::unordered_set<hnswlib::labeltype> khop_nbrs = grs.getKHopNodes(query_label, k_hop);
            KHopFilter hnswbf_filter(khop_nbrs);
            auto approx_results_filtered = normal_index->searchKnn(query_vector, k_query, &hnswbf_filter);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto approx_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            sum_us += approx_time_us;

            // 结果必须在 k-hop 内
            verifyResults(approx_results_filtered, khop_nbrs, query_label, "带过滤器HNSW");

            // 计算召回率（与真值 Top-k 交集）
            int matches = 0;
            auto approx_copy = approx_results_filtered;
            while (!approx_copy.empty()) {
                if (exact_label_sets[qi].count(approx_copy.top().second) > 0) {
                    matches++;
                }
                approx_copy.pop();
            }
            sum_recall += static_cast<double>(matches) / static_cast<double>(k_query);
        }

        double avg_recall = sum_recall / static_cast<double>(num_queries);
        int avg_us = static_cast<int>(std::llround(static_cast<double>(sum_us) / num_queries)); // 平均微秒，四舍五入
        recall_us_pairs.emplace_back(avg_recall * 100.0, avg_us);
        std::cout << "ef=" << ef << "  avg_recall=" << std::fixed << std::setprecision(3)
                  << (avg_recall * 100.0) << "%  avg_time=" << avg_us << " us\n";
    }

    std::ostringstream oss;
    oss << "Benchmark Report\n";
    oss << "Search Times (ns):\n";
    oss << "Index \\ ef |";
    for (size_t ef : efs) oss << " ef=" << ef;
    oss << "\n";

    // 仅第一条含 (recall, us) 的行会被 parse_search_times 解析
    oss << "HNSW: ";
    oss << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < recall_us_pairs.size(); ++i) {
        oss << "(" << recall_us_pairs[i].first << ", " << recall_us_pairs[i].second << " us)";
        if (i + 1 < recall_us_pairs.size()) oss << " ";
    }
    oss << "\n\n----------------------------------------\n";

    std::ofstream fout((log_dir + "/HNSW.log").c_str(), std::ios::out | std::ios::trunc);
    fout << oss.str();
    fout.close();
    std::cout << "日志已写入: " << (log_dir + "/HNSW.log") << std::endl;

    // =================== HNSW 评测结束 ===================

    // ==================== HNSW+PSL 评测 ====================
    std::cout << "正在构建HNSW+PSL..." << std::endl;
    build_start = std::chrono::high_resolution_clock::now();
    hnswlib::HierarchicalNSW<float>* hnsw_psl_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    for (int i = 0; i < max_elements; i++) {
        hnsw_psl_index->addPoint(data + i * dim, i);
    }
    hnswlib::DisOracle psl_index(grs.edge_pairs, k_hop, false);
    psl_index.see_labels();
    build_end = std::chrono::high_resolution_clock::now();
    auto hnswpsl_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "HNSW+PSL构建完成，耗时: " << hnswpsl_build_time << " 毫秒" << std::endl;
    std::cout << "HNSW+PSL: HNSW index size: " << hnsw_psl_index->indexFileSize() << std::endl;


    std::vector<std::pair<double,int>> ghnsw_recall_us_pairs;

    for (size_t ef : efs) {
        normal_index->setEf(ef);

        long long sum_us = 0;
        double sum_recall = 0.0;

        for (int qi = 0; qi < num_queries; ++qi) {
            int query_idx = query_ids[qi];
            float* query_vector = data + query_idx * dim;
            hnswlib::labeltype query_label = query_idx;

            // PSL 过滤器
            PSLFilter hnswpsl_filter(query_label, k_hop, &psl_index);

            auto t1 = std::chrono::high_resolution_clock::now();
            auto approx_results_psl = normal_index->searchKnn(query_vector, k_query, &hnswpsl_filter);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto approx_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            sum_us += approx_time_us;

            // // 校验结果均在 k-hop 内
            // std::unordered_set<hnswlib::labeltype> khop_nbrs = grs.getKHopNodes(query_label, k_hop);
            // verifyResults(approx_results_psl, khop_nbrs, query_label, "HNSW+PSL");

            // 召回率：与对应真值集合求交
            int matches = 0;
            auto approx_copy = approx_results_psl;
            while (!approx_copy.empty()) {
                if (exact_label_sets[qi].count(approx_copy.top().second) > 0) {
                    matches++;
                }
                approx_copy.pop();
            }
            sum_recall += static_cast<double>(matches) / static_cast<double>(k_query);
        }

        double avg_recall = sum_recall / static_cast<double>(num_queries);
        int avg_us = static_cast<int>(std::llround(static_cast<double>(sum_us) / num_queries));
        ghnsw_recall_us_pairs.emplace_back(avg_recall * 100.0, avg_us);
        std::cout << "[GHNSW] ef=" << ef << "  avg_recall=" << std::fixed << std::setprecision(3)
                  << (avg_recall * 100.0) << "%  avg_time=" << avg_us << " us\n";
    }

    // 写 GHNSW 日志（供 parse_search_times 使用）
    std::ostringstream oss_ghnsw;
    oss_ghnsw << "Benchmark Report\n";
    oss_ghnsw << "Search Times (ns):\n";
    oss_ghnsw << "Index \\ ef |";
    for (size_t ef : efs) oss_ghnsw << " ef=" << ef;
    oss_ghnsw << "\n";

    oss_ghnsw << "GHNSW: ";
    oss_ghnsw << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < ghnsw_recall_us_pairs.size(); ++i) {
        oss_ghnsw << "(" << ghnsw_recall_us_pairs[i].first << ", " << ghnsw_recall_us_pairs[i].second << " us)";
        if (i + 1 < ghnsw_recall_us_pairs.size()) oss_ghnsw << " ";
    }
    oss_ghnsw << "\n\n----------------------------------------\n";

    std::ofstream fout_ghnsw((log_dir + "/GHNSW.log").c_str(), std::ios::out | std::ios::trunc);
    fout_ghnsw << oss_ghnsw.str();
    fout_ghnsw.close();
    std::cout << "日志已写入: " << (log_dir + "/GHNSW.log") << std::endl;
    // ================== HNSW+PSL 评测结束 ===================

    // ==================== ACORN 评测 ====================

    // ACORN 参数
    int acorn_M = 64;
    int acorn_gamma = 1;
    int acorn_M_beta = 128;
    std::cout << "\n正在构建ACORN索引..." << std::endl;
    auto acorn_build_start = std::chrono::high_resolution_clock::now();

    // 为 ACORN 准备元数据（简单地为每个点分配一个类别标签）
    std::vector<int> acorn_metadata(max_elements);
    for (int i = 0; i < max_elements; ++i) {
        acorn_metadata[i] = i ;
    }

    faiss::IndexACORNFlat acorn_index(dim, acorn_M, acorn_gamma, acorn_metadata, acorn_M_beta, faiss::METRIC_L2);
    acorn_index.add(max_elements, data);

    auto acorn_build_end = std::chrono::high_resolution_clock::now();
    auto acorn_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(acorn_build_end - acorn_build_start).count();
    std::cout << "ACORN索引构建完成，耗时: " << acorn_build_ms << " 毫秒" << std::endl;

    // 评测（与 HNSW 使用相同 ef 列表与相同 k-hop 过滤）
    std::vector<std::pair<double,int>> acorn_recall_us_pairs;

    for (size_t ef : efs) {
        long long acorn_sum_us = 0;
        double acorn_sum_recall = 0.0;

        // 设置 ACORN 搜索参数（efSearch）
        acorn_index.acorn.efSearch = static_cast<int>(ef);

        for (int qi = 0; qi < num_queries; ++qi) {
            int query_idx = query_ids[qi];
            float* query_vector = data + query_idx * dim;
            hnswlib::labeltype query_label = query_idx;

            std::vector<faiss::idx_t> acorn_labels(k_query);
            std::vector<float> acorn_dist(k_query);
            std::vector<char> acorn_filter_one(max_elements);

            auto t1 = std::chrono::high_resolution_clock::now();
            // 构建该查询的过滤映射（k-hop 内为 1）
            std::unordered_set<hnswlib::labeltype> khop_nbrs = grs.getKHopNodes(query_label, k_hop);
            for (auto id : khop_nbrs) acorn_filter_one[id] = 1;
            acorn_filter_one[query_label] = 1;
            
            acorn_index.search(
                1,                         // nq
                query_vector,              // x
                k_query,                   // k
                acorn_dist.data(),
                acorn_labels.data(),
                acorn_filter_one.data()
            );
            auto t2 = std::chrono::high_resolution_clock::now();
            auto acorn_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            acorn_sum_us += acorn_time_us;

            // // 验证结果在 k-hop 内
            // for (int r = 0; r < k_query; ++r) {
            //     auto lid = acorn_labels[r];
            //     if (lid >= 0) {
            //         if (lid != query_label && khop_nbrs.count(static_cast<hnswlib::labeltype>(lid)) == 0) {
            //             std::cerr << "Error: ACORN 结果 " << lid << " 不在查询点 " << query_label << " 的 k-hop 邻居内\n";
            //             assert(false);
            //         }
            //     }
            // }

            // 计算召回
            int matches = 0;
            for (int r = 0; r < k_query; ++r) {
                auto lid = acorn_labels[r];
                if (lid >= 0 && exact_label_sets[qi].count(static_cast<hnswlib::labeltype>(lid)) > 0) {
                    matches++;
                }
            }
            acorn_sum_recall += static_cast<double>(matches) / static_cast<double>(k_query);
        }

        double acorn_avg_recall = acorn_sum_recall / static_cast<double>(num_queries);
        int acorn_avg_us = static_cast<int>(std::llround(static_cast<double>(acorn_sum_us) / num_queries));
        acorn_recall_us_pairs.emplace_back(acorn_avg_recall * 100.0, acorn_avg_us);
        std::cout << "[ACORN] ef=" << ef << "  avg_recall=" << std::fixed << std::setprecision(3)
                  << (acorn_avg_recall * 100.0) << "%  avg_time=" << acorn_avg_us << " us\n";
    }

    // 写 ACORN 日志
    std::ostringstream oss_acorn;
    oss_acorn << "Benchmark Report\n";
    oss_acorn << "Search Times (ns):\n";
    oss_acorn << "Index \\ ef |";
    for (size_t ef : efs) oss_acorn << " ef=" << ef;
    oss_acorn << "\n";

    oss_acorn << "ACORN: ";
    oss_acorn << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < acorn_recall_us_pairs.size(); ++i) {
        oss_acorn << "(" << acorn_recall_us_pairs[i].first << ", " << acorn_recall_us_pairs[i].second << " us)";
        if (i + 1 < acorn_recall_us_pairs.size()) oss_acorn << " ";
    }
    oss_acorn << "\n\n----------------------------------------\n";

    std::ofstream fout_acorn((log_dir + "/ACORN.log").c_str(), std::ios::out | std::ios::trunc);
    fout_acorn << oss_acorn.str();
    fout_acorn.close();
    std::cout << "日志已写入: " << (log_dir + "/ACORN.log") << std::endl;

    // ==================== ACORN 评测结束 ====================

    // 在main函数结束前调用
    analyzeKHopDistribution(grs, max_elements, 5); // 分析1到5跳的邻居分布
    
    return 0;
}
