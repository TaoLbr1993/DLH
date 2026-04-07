#include "../../hnswlib/hnswlib.h"
#include "../common/dataset_io.h"
#include "../common/metrics.h"
#include "faiss/index_io.h"
#include <chrono>
#include <faiss/IndexHNSW.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

static void split_csv(const std::string &s, std::vector<size_t> &out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            out.push_back((size_t)std::stoul(tok));
}

int main(int argc, char **argv) {
    auto t_program_start = std::chrono::steady_clock::now();

    if (argc < 3) {
        std::cerr << "用法: baseline_navix --data-dir DIR --out OUT_DIR "
                     "--ef-list 20,50,100 [--M 32 --efC 200]\n";
        return 1;
    }
    std::string data_dir, out_dir, ef_list_str;
    int M = 32, efC = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&] { return std::string(argv[++i]); };
        if (a == "--data-dir")
            data_dir = nxt();
        else if (a == "--out")
            out_dir = nxt();
        else if (a == "--ef-list")
            ef_list_str = nxt();
        else if (a == "--M")
            M = std::stoi(nxt());
        else if (a == "--efC")
            efC = std::stoi(nxt());
    }
    if (data_dir.empty() || out_dir.empty() || ef_list_str.empty()) {
        std::cerr << "缺少必要参数\n";
        return 1;
    }

    std::cout << "[参数] --data-dir=" << data_dir
              << " --out=" << out_dir
              << " --ef-list=\"" << ef_list_str << "\""
              << " --M=" << M
              << " --efC=" << efC
              << std::endl;

    auto ds = load_dataset_all(data_dir);

    // 打印 metric
    std::cout << "[信息] dataset metric=" << ds.meta.metric << "\n";

    // cosine：需要归一化（确保与 groundtruth 一致）
    if (ds.meta.metric == "cosine") {
        std::cout << "[信息] metric=cosine，对 base 向量做 L2 归一化...\n";
        l2_normalize_dataset_inplace(ds.base, ds.meta.dim);
    } else if (ds.meta.metric != "l2" && !ds.meta.metric.empty()) {
        std::cout << "[警告] 未识别的 metric=" << ds.meta.metric << "，将按 l2 处理\n";
    }

    // 若没有映射文件
    if (ds.orig2super.empty()) {
        std::cerr << "[信息] 未发现超级节点映射文件\n";
        return 1;
    } else {
        std::cout << "[信息] 已加载超级节点映射: supernodes=" << ds.meta.supernodes
                  << " group_size=" << ds.meta.map_group_size << "\n";
    }


    hnswlib::GraphRelationSampler grs(ds.meta.prob);
    if (!load_graph(grs, ds.meta.graph_file))
        return 1;

    // gt sets
    std::vector<std::unordered_set<int>> gt_sets(ds.queries.Q);
    for (int qi = 0; qi < ds.queries.Q; ++qi) {
        auto *beg = ds.gt_labels.data() + (size_t)qi * ds.K;
        gt_sets[qi] = std::unordered_set<int>(beg, beg + ds.K);
    }

    auto t_build_start = std::chrono::steady_clock::now();

    // 构建 Faiss HNSW-Flat（按 metric 选择距离类型：cosine=归一化后用内积）
    faiss::MetricType faiss_metric =
        (ds.meta.metric == "cosine") ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    faiss::IndexHNSWFlat navix(ds.meta.dim, M, faiss_metric);
    navix.hnsw.efConstruction = efC;
    navix.add(ds.meta.N_used, ds.base.data());

    auto t_build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count();
    std::cout << "[计时] 索引构建时间: " << build_ms << " ms" << std::endl;


    auto t_eval_start = std::chrono::steady_clock::now();

    std::vector<size_t> efs;
    split_csv(ef_list_str, efs);
    std::vector<std::pair<double, int>> recall_us_pairs;

    // 临时缓冲
    std::vector<float> dist(ds.meta.topk);
    std::vector<faiss::idx_t> labels(ds.meta.topk);

    for (size_t ef : efs) {
        navix.hnsw.efSearch = (int)ef;

        long long sum_us = 0;
        double sum_recall = 0.0;

        for (int qi = 0; qi < ds.queries.Q; ++qi) {
            if (qi % 1000 == 0 && qi > 0) {
                std::cout << "  查询进度: " << qi << " / " << ds.queries.Q << "\r"
                          << std::flush;
            }
            
            int qid = ds.queries.flat[qi * 2 + 0];
            int hopi = ds.queries.flat[qi * 2 + 1];
            const float *q = ds.base.data() + (size_t)qid * ds.meta.dim;

            auto t1 = std::chrono::high_resolution_clock::now();

            // 超级节点 k-hop 过滤：原始查询 id -> 超级节点 -> k-hop 超级节点集合 -> 展开到原始 id
            int q_super = ds.orig2super[(size_t)qid];
            auto khop_supers = grs.getKHopNodes(q_super, hopi);

            std::vector<char> mask(ds.meta.N_used, 0);
            for (auto sid : khop_supers) {
                const auto &members = ds.super_members[sid];
                for (int oid : members) mask[(size_t)oid] = 1;
            }

            faiss::VisitedTable vt(ds.meta.N_used);
            faiss::HNSWStats stats;
            
            navix.navix_single_search(q, ds.meta.topk, dist.data(),
                                      labels.data(), mask.data(), vt, stats);
            auto t2 = std::chrono::high_resolution_clock::now();
            sum_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                    .count();

            std::vector<int> approx(ds.meta.topk, -1);
            for (int i = 0; i < ds.meta.topk; ++i)
                approx[i] = (int)labels[i];
            sum_recall += recall_at_k_from_sets(approx, gt_sets[qi]);
        }

        int avg_us = (int)std::llround((double)sum_us / (double)ds.queries.Q);
        double avg_rec = sum_recall / (double)ds.queries.Q;
        recall_us_pairs.emplace_back(avg_rec, avg_us);
        std::cout << "[NAVIX] ef=" << ef << " recall=" << std::fixed
                  << std::setprecision(3) << avg_rec * 100.0
                  << "% time=" << avg_us << " us\n";
    }

    auto t_eval_end = std::chrono::steady_clock::now();
    auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_eval_end - t_eval_start).count();
    std::cout << "[计时] 评测时间(包含全部ef轮次): " << eval_ms << " ms" << std::endl;

    std::ostringstream oss;
    oss << "Benchmark Report\n";
    oss << "Search Times (ns):\n";
    oss << "Index \\ ef |";
    for (auto ef : efs)
        oss << " ef=" << ef;
    oss << "\nNAVIX: " << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < recall_us_pairs.size(); ++i) {
        oss << "(" << recall_us_pairs[i].first * 100.0 << ", "
            << recall_us_pairs[i].second << " us)";
        if (i + 1 < recall_us_pairs.size())
            oss << " ";
    }
    oss << "\n";
    std::ofstream fout(out_dir + "/NAVIX_stats.log", std::ios::out | std::ios::trunc);
    fout << oss.str();
    std::cout << "日志写入: " << (out_dir + "/NAVIX_stats.log") << std::endl;

    auto t_program_end = std::chrono::steady_clock::now();
    auto program_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_program_end - t_program_start).count();
    std::cout << "[计时] 程序总运行时间: " << program_ms << " ms" << std::endl;

    return 0;
}