#include "../../hnswlib/hnswlib.h"
#include "../common/dataset_io.h"
#include "../common/metrics.h"

#include <faiss/IndexACORN.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>

static void split_csv(const std::string &s, std::vector<size_t> &out) {
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back((size_t)std::stoul(tok));
    }
}

int main(int argc, char **argv) {
    auto t_program_start = std::chrono::steady_clock::now();

    if (argc < 3) {
        std::cerr << "用法: baseline_acorn --data-dir DIR --out OUT_DIR "
                     "--ef-list 20,50,100 [--acorn-M 64 --acorn-gamma 1 --acorn-M-beta 128]\n";
        return 1;
    }

    std::string data_dir, out_dir, ef_list_str;
    int acorn_M = 32, acorn_gamma = 2, acorn_M_beta = 64;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&] { return std::string(argv[++i]); };
        if (a == "--data-dir") data_dir = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--ef-list") ef_list_str = nxt();
        else if (a == "--acorn-M") acorn_M = std::stoi(nxt());
        else if (a == "--acorn-gamma") acorn_gamma = std::stoi(nxt());
        else if (a == "--acorn-M-beta") acorn_M_beta = std::stoi(nxt());
    }
    if (data_dir.empty() || out_dir.empty() || ef_list_str.empty()) {
        std::cerr << "缺少必要参数\n";
        return 1;
    }

    std::cout << "[参数] --data-dir=" << data_dir
              << " --out=" << out_dir
              << " --ef-list=\"" << ef_list_str << "\""
              << " --acorn-M=" << acorn_M
              << " --acorn-gamma=" << acorn_gamma
              << " --acorn-M-beta=" << acorn_M_beta
              << std::endl;


    // 载入数据
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


    // 载入图（用于构建 k-hop 过滤）
    hnswlib::GraphRelationSampler grs(ds.meta.prob);
    if (!load_graph(grs, ds.meta.graph_file))
        return 1;

    // groundtruth -> 每个查询的集合
    std::vector<std::unordered_set<int>> gt_sets(ds.queries.Q);
    for (int qi = 0; qi < ds.queries.Q; ++qi) {
        auto *beg = ds.gt_labels.data() + (size_t)qi * ds.K;
        gt_sets[qi] = std::unordered_set<int>(beg, beg + ds.K);
    }

    auto t_build_start = std::chrono::steady_clock::now();

    // 构建 ACORN 索引
    std::vector<int> acorn_metadata(ds.meta.N_used);
    for (size_t i = 0; i < ds.meta.N_used; ++i) acorn_metadata[i] = (int)i;

    // 按 metric 选择 Faiss 距离类型：cosine=归一化后用内积
    faiss::MetricType faiss_metric =
        (ds.meta.metric == "cosine") ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    faiss::IndexACORNFlat acorn_index(
        (int)ds.meta.dim,
        acorn_M,
        acorn_gamma,
        acorn_metadata,
        acorn_M_beta,
        faiss_metric);

    acorn_index.add((faiss::idx_t)ds.meta.N_used, ds.base.data());

    auto t_build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count();
    std::cout << "[计时] 索引构建时间: " << build_ms << " ms" << std::endl;

    // ef 列表
    std::vector<size_t> efs;
    split_csv(ef_list_str, efs);


    auto t_eval_start = std::chrono::steady_clock::now();

    // 评测
    std::vector<std::pair<double, int>> recall_us_pairs;

    std::vector<faiss::idx_t> labels(ds.meta.topk);
    std::vector<float> dists(ds.meta.topk);
    std::vector<char> filter(ds.meta.N_used, 0);

    for (size_t ef : efs) {
        acorn_index.acorn.efSearch = (int)ef;

        long long sum_us = 0;
        double sum_recall = 0.0;

        for (int qi = 0; qi < ds.queries.Q; ++qi) {
            if (qi % 1000 == 0 && qi > 0) {
                std::cout << "  查询进度: " << qi << " / " << ds.queries.Q << "\r" << std::flush;
            }

            int qid  = ds.queries.flat[qi * 2 + 0];
            int hopi = ds.queries.flat[qi * 2 + 1];
            const float *q = ds.base.data() + (size_t)qid * ds.meta.dim;

            auto t1 = std::chrono::high_resolution_clock::now();

            // 超级节点 k-hop 过滤：查询原始 id -> 超级节点 -> k-hop 超级节点集合 -> 展开至原始 id
            std::fill(filter.begin(), filter.end(), 0);
            int q_super = ds.orig2super[(size_t)qid];
            auto khop_supers = grs.getKHopNodes(q_super, hopi);
            for (auto sid : khop_supers) {
                const auto &members = ds.super_members[sid];
                for (int oid : members) {
                    if ((size_t)oid < ds.meta.N_used) filter[(size_t)oid] = 1;
                }
            }
            // // 保证查询自身可见
            // if ((size_t)qid < ds.meta.N_used) filter[(size_t)qid] = 1;

            acorn_index.search(1, q, (faiss::idx_t)ds.meta.topk, dists.data(), labels.data(), filter.data());
            auto t2 = std::chrono::high_resolution_clock::now();
            sum_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            // 按输出顺序收集 labels（已按距离升序）
            std::vector<int> approx(ds.meta.topk, -1);
            for (int k = 0; k < (int)ds.meta.topk; ++k) {
                approx[k] = (int)labels[k];
            }
            sum_recall += recall_at_k_from_sets(approx, gt_sets[qi]);
        }

        int avg_us = (int)std::llround((double)sum_us / (double)ds.queries.Q);
        double avg_recall = sum_recall / (double)ds.queries.Q;
        // 存原始 recall（0~1）
        recall_us_pairs.emplace_back(avg_recall, avg_us);
        std::cout << "[ACORN] ef=" << ef << " recall=" << std::fixed
                  << std::setprecision(3) << avg_recall * 100.0
                  << "% time=" << avg_us << " us\n";
    }

    auto t_eval_end = std::chrono::steady_clock::now();
    auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_eval_end - t_eval_start).count();
    std::cout << "[计时] 评测时间(包含全部ef轮次): " << eval_ms << " ms" << std::endl;

    // 写日志（与 parse_search_times 兼容）
    std::ostringstream oss;
    oss << "Benchmark Report\n";
    oss << "Search Times (ns):\n";
    oss << "Index \\ ef |";
    for (auto ef : efs) oss << " ef=" << ef;
    oss << "\nACORN: " << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < recall_us_pairs.size(); ++i) {
        oss << "(" << recall_us_pairs[i].first * 100.0 << ", "
            << recall_us_pairs[i].second << " us)";
        if (i + 1 < recall_us_pairs.size()) oss << " ";
    }
    oss << "\n";

    std::ofstream fout(out_dir + "/ACORN_stats.log", std::ios::out | std::ios::trunc);
    fout << oss.str();
    std::cout << "日志写入: " << (out_dir + "/ACORN_stats.log") << std::endl;

    auto t_program_end = std::chrono::steady_clock::now();
    auto program_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_program_end - t_program_start).count();
    std::cout << "[计时] 程序总运行时间: " << program_ms << " ms" << std::endl;

    return 0;
}