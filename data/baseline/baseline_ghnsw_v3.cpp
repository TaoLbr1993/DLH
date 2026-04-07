#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/pslV3.h"
#include "../common/dataset_io.h"
#include "../common/metrics.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <sys/stat.h>

// 新增：公共内存统计
#include "../common/memstat.h"

// PSL 过滤器
// 原版本基于原始点 id，现在改为基于超级节点 id 进行 PSL 过滤：
// 1) 查询点原始 id -> q_super_
// 2) 候选点原始 id -> cand_super
// 3) PSL 使用超级节点对进行 hop 判断
class PSLFilter : public hnswlib::BaseFilterFunctor {
  public:
    PSLFilter(int query_orig_id, int hop, DisOracle* oracle, const std::vector<int>* orig2super)
        : hop_(hop), psl_(oracle), map_(orig2super) {
        q_super_ = (*map_)[(size_t)query_orig_id];
    }
    bool operator()(hnswlib::labeltype cand_orig_id) override {
        int cand_super = (*map_)[(size_t)cand_orig_id];
        return psl_->query(q_super_, cand_super, hop_);
    }
  private:
    int q_super_;
    int hop_;
    DisOracle* psl_;
    const std::vector<int>* map_;
};

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
        std::cerr << "用法: baseline_hnsw_psl --data-dir DIR --out OUT_DIR "
                     "--ef-list 20,50,100 [--M 16 --efC 200 --bf-fpp 0.01]\n";
        return 1;
    }
    std::string data_dir, out_dir, ef_list_str;
    int M = 16, efC = 200;
    double bf_fpp = 0.01; // 新增：布隆过滤器的 FPP

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&] { return std::string(argv[++i]); };
        if (a == "--data-dir") data_dir = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--ef-list") ef_list_str = nxt();
        else if (a == "--M") M = std::stoi(nxt());
        else if (a == "--efC") efC = std::stoi(nxt());
        else if (a == "--bf-fpp") bf_fpp = std::stod(nxt()); // 新增
    }
    if (data_dir.empty() || out_dir.empty() || ef_list_str.empty()) {
        std::cerr << "缺少必要参数\n";
        return 1;
    }

    // 合法性检查与夹取
    if (!(bf_fpp > 0.0 && bf_fpp < 1.0)) {
        std::cerr << "[警告] --bf-fpp 非法，使用默认 0.01\n";
        bf_fpp = 0.01;
    }

    // 新增：打印参数
    std::cout << "[参数] --data-dir=" << data_dir
              << " --out=" << out_dir
              << " --ef-list=\"" << ef_list_str << "\""
              << " --M=" << M
              << " --efC=" << efC
              << " --bf-fpp=" << bf_fpp
              << std::endl;

    // 新增：所有 baseline 复用的内存日志器
    memstat::Logger memlog;
    memlog.snap("程序启动");

    // 载入数据
    auto ds = load_dataset_all(data_dir);

    // 建议：打印 metric（如果 meta 里有）
    std::cout << "[信息] dataset metric=" << ds.meta.metric << "\n";

    // cosine：需要归一化（确保与 groundtruth 一致）
    if (ds.meta.metric == "cosine") {
        std::cout << "[信息] metric=cosine，对 base 向量做 L2 归一化...\n";
        l2_normalize_dataset_inplace(ds.base, ds.meta.dim);
    }

    if (ds.orig2super.empty()) {
        std::cerr << "[信息] 未检测到超级节点映射\n";
        return 1;
    } else {
        std::cout << "[信息] 已加载超级节点映射: supernodes=" << ds.meta.supernodes
                  << " group_size=" << ds.meta.map_group_size << "\n";
    }

    // 载入图
    hnswlib::GraphRelationSampler grs(ds.meta.prob);
    if (!load_graph(grs, ds.meta.graph_file))
        return 1;

    // groundtruth -> 每个查询的集合
    std::vector<std::unordered_set<int>> gt_sets(ds.queries.Q);
    for (int qi = 0; qi < ds.queries.Q; ++qi) {
        auto *beg = ds.gt_labels.data() + (size_t)qi * ds.K;
        gt_sets[qi] = std::unordered_set<int>(beg, beg + ds.K);
    }

    memlog.snap("数据载入完成");

    // 新增：统计索引构建时间
    auto t_build_start = std::chrono::steady_clock::now();

    // 构建 HNSW（按 metric 选择 space）
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    if (ds.meta.metric == "cosine") {
        space.reset(new hnswlib::InnerProductSpace(ds.meta.dim));
    } else {
        space.reset(new hnswlib::L2Space(ds.meta.dim));
    }

    hnswlib::HierarchicalNSW<float> index(space.get(), ds.meta.N_used, M, efC);
    for (size_t i = 0; i < ds.meta.N_used; ++i)
        index.addPoint(ds.base.data() + i * ds.meta.dim, i);


    // 构建 PSL（基于超级节点图的边）
    DisOracle psl_oracle(grs.edge_pairs, ds.meta.k_hop_max, false);
    psl_oracle.see_labels(); // 与示例保持一致

    // std::cout << "BFLabelHashEle::projected_element_divisor = " << std::fixed << std::setprecision(3)
    //     << BFLabelHashEle::projected_element_divisor << std::endl;
    
    auto t_build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count();
    std::cout << "[计时] 索引构建时间: " << build_ms << " ms" << std::endl;

    memlog.snap("索引构建完成");

    // 打印索引大小
    std::cout << "HNSW 索引大小: " << index.indexFileSize() << " bytes" << std::endl;

    // 只需要拿到大小信息，直接返回
    return 0;

    // ef 列表
    std::vector<size_t> efs;
    split_csv(ef_list_str, efs);

    // 新增：评测计时起点
    auto t_eval_start = std::chrono::steady_clock::now();

    // 评测
    std::vector<std::pair<double, int>> recall_us_pairs;

    for (size_t ef : efs) {
        index.setEf(ef);
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
            // 使用超级节点 id 初始化查询
            int q_super = ds.orig2super[(size_t)qid];
            // psl_oracle.init_query_node(q_super);
            PSLFilter filter(qid, hopi, &psl_oracle, &ds.orig2super);

            auto res = index.searchKnn((void *)q, ds.meta.topk, &filter);
            auto t2 = std::chrono::high_resolution_clock::now();
            sum_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            // 收集 labels（按距离升序）
            std::vector<int> approx(ds.meta.topk, -1);
            int cnt = 0;
            while (!res.empty() && cnt < ds.meta.topk) {
                approx[ds.meta.topk - 1 - cnt] = (int)res.top().second;
                res.pop();
                cnt++;
            }
            sum_recall += recall_at_k_from_sets(approx, gt_sets[qi]);
        }

        int avg_us = (int)std::llround((double)sum_us / (double)ds.queries.Q);
        double avg_recall = sum_recall / (double)ds.queries.Q;
        // 存原始 recall（0~1）
        recall_us_pairs.emplace_back(avg_recall, avg_us);
        std::cout << "[GHNSW] ef=" << ef << " recall=" << std::fixed
                  << std::setprecision(3) << avg_recall * 100.0
                  << "% time=" << avg_us << " us\n";
    }

    // 新增：评测计时终点并打印
    auto t_eval_end = std::chrono::steady_clock::now();
    auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_eval_end - t_eval_start).count();
    std::cout << "[计时] 评测时间(包含全部ef轮次): " << eval_ms << " ms" << std::endl;

    memlog.snap("评测完成");

    // 写日志（与 parse_search_times 兼容）
    std::ostringstream oss;
    oss << "Benchmark Report\n";
    oss << "Search Times (ns):\n";
    oss << "Index \\ ef |";
    for (auto ef : efs) oss << " ef=" << ef;
    oss << "\nGHNSW: " << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < recall_us_pairs.size(); ++i) {
        oss << "(" << recall_us_pairs[i].first * 100.0 << ", "
            << recall_us_pairs[i].second << " us)";
        if (i + 1 < recall_us_pairs.size()) oss << " ";
    }
    oss << "\n";

    std::ofstream fout(out_dir + "/GHNSW-V3_stats.log", std::ios::out | std::ios::trunc);
    fout << oss.str();
    std::cout << "日志写入: " << (out_dir + "/GHNSW-V3_stats.log") << std::endl;

    // 新增：程序总运行时间
    auto t_program_end = std::chrono::steady_clock::now();
    auto program_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_program_end - t_program_start).count();
    std::cout << "[计时] 程序总运行时间: " << program_ms << " ms" << std::endl;

    memlog.snap("程序结束");

    memlog.dump_to_file(out_dir + "/GHNSW-V3_memstat.log");

    return 0;
}