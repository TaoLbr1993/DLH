#include "../../hnswlib/hnswlib.h"
#include "../common/dataset_io.h"
#include "../common/metrics.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>


class KHopFilter : public hnswlib::BaseFilterFunctor {
  public:
    KHopFilter(const std::unordered_set<hnswlib::labeltype>& nbrs) : nbrs_(nbrs) {}
    bool operator()(hnswlib::labeltype id) override { return nbrs_.count(id) > 0; }
  private:
    const std::unordered_set<hnswlib::labeltype>& nbrs_;
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
        std::cerr << "用法: baseline_bf --data-dir DIR --out OUT_DIR --ef-list 20,50,100\n";
        return 1;
    }

    std::string data_dir, out_dir, ef_list_str;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&] { return std::string(argv[++i]); };
        if (a == "--data-dir") data_dir = nxt();
        else if (a == "--out") out_dir = nxt();
        else if (a == "--ef-list") ef_list_str = nxt();
    }
    if (data_dir.empty() || out_dir.empty() || ef_list_str.empty()) {
        std::cerr << "缺少必要参数\n";
        return 1;
    }

    std::cout << "[参数] --data-dir=" << data_dir
              << " --out=" << out_dir
              << " --ef-list=\"" << ef_list_str << "\""
              << std::endl;

    // 载入数据与图
    auto ds = load_dataset_all(data_dir);
    hnswlib::GraphRelationSampler grs(ds.meta.prob);
    if (!load_graph(grs, ds.meta.graph_file))
        return 1;

    // GT -> 集合
    std::vector<std::unordered_set<int>> gt_sets(ds.queries.Q);
    for (int qi = 0; qi < ds.queries.Q; ++qi) {
        auto *beg = ds.gt_labels.data() + (size_t)qi * ds.K;
        gt_sets[qi] = std::unordered_set<int>(beg, beg + ds.K);
    }

    // 构建 BF 索引
    auto t_build_start = std::chrono::steady_clock::now();
    hnswlib::L2Space space(ds.meta.dim);
    hnswlib::BruteforceSearch<float> bf(&space, ds.meta.N_used);
    for (size_t i = 0; i < ds.meta.N_used; ++i) {
        bf.addPoint(ds.base.data() + i * ds.meta.dim, i);
    }
    auto t_build_end = std::chrono::steady_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count();
    std::cout << "[计时] 索引构建时间(BF): " << build_ms << " ms" << std::endl;

    // 评测（一次，复用到所有 ef）
    auto t_eval_start = std::chrono::steady_clock::now();
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
        // 获取 k-hop 邻居集合
        auto khop_nbrs = grs.getKHopNodes(qid, hopi);
        KHopFilter filter(khop_nbrs);
        
        auto res = bf.searchKnn((void *)q, ds.meta.topk, &filter);
        auto t2 = std::chrono::high_resolution_clock::now();
        sum_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

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

    auto t_eval_end = std::chrono::steady_clock::now();
    auto eval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_eval_end - t_eval_start).count();
    std::cout << "[BF] recall=" << std::fixed << std::setprecision(3) << avg_recall * 100.0
              << "% time=" << avg_us << " us" << std::endl;
    std::cout << "[计时] 评测时间: " << eval_ms << " ms" << std::endl;

    // ef 列表
    std::vector<size_t> efs;
    split_csv(ef_list_str, efs);

    // 写日志
    std::ostringstream oss;
    oss << "Benchmark Report\n";
    oss << "Search Times (ns):\n";
    oss << "Index \\ ef |";
    for (auto ef : efs) oss << " ef=" << ef;
    oss << "\nBF: " << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < efs.size(); ++i) {
        oss << "(" << avg_recall * 100.0 << ", " << avg_us << " us)";
        if (i + 1 < efs.size()) oss << " ";
    }
    oss << "\n";

    std::ofstream fout(out_dir + "/BF_stats.log", std::ios::out | std::ios::trunc);
    fout << oss.str();
    std::cout << "日志写入: " << (out_dir + "/BF_stats.log") << std::endl;

    auto t_program_end = std::chrono::steady_clock::now();
    auto program_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_program_end - t_program_start).count();
    std::cout << "[计时] 程序总运行时间: " << program_ms << " ms" << std::endl;
    return 0;
}