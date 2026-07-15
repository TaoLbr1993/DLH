#include "../hnswlib/hnswlib.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <queue>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <iomanip>
#include <cmath>
// ========================================
// 参数（默认值，支持命令行覆盖）
static std::string fvecs_path = "/home/jiangyuntao/Sift1M/sift_base.fvecs";
static int max_elements = 200000; // 只读前 N 条记录
static std::string output_dir = "/home/jiangyuntao/test_data/Sift1M-0.0003-20w-3hop";
static float prob = 0.0003f;      // 概率图建边概率
static int topk = 10;             // Top-K
static int k_query = 1000;        // 查询数量
static uint64_t random_seed = 47; // 随机种子
static int k_hop_min = 3;         // 每个 query 随机选择 k-hop 的范围（含端点）
static int k_hop_max = 3;
// 新增映射参数
static int map_group_size = 2; // 每 group_size 个原始向量映射为一个超级节点
static std::string metric = "l2"; // 新增：距离度量 l2 / cosine
static std::string graph_model = "er"; // er / lfr
static int lfr_avg_degree = 15;
static int lfr_max_degree = 30;
static double lfr_degree_tau = 2.5;
static double lfr_community_tau = 1.5;
static double lfr_mu = 0.3;
static size_t lfr_min_community_size = 20;
static size_t lfr_max_community_size = 1000;
// ========================================

// 读取 fvecs 的维度与记录数
static void read_fvecs_meta(const std::string& path, int& d, size_t& n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { std::cerr << "无法打开 fvecs 文件: " << path << std::endl; std::exit(1); }
    int dim = 0;
    fin.read(reinterpret_cast<char*>(&dim), 4);
    if (!fin) { std::cerr << "读取维度失败: " << path << std::endl; std::exit(1); }
    fin.seekg(0, std::ios::end);
    std::streampos fsize = fin.tellg();
    if (fsize <= 0) { std::cerr << "文件大小异常: " << path << std::endl; std::exit(1); }
    const size_t rec_bytes = 4 + static_cast<size_t>(dim) * 4;
    if (rec_bytes == 0 || static_cast<size_t>(fsize) % rec_bytes != 0) {
        std::cerr << "文件尺寸与记录长度不匹配，可能不是标准 fvecs: " << path << std::endl; std::exit(1);
    }
    n = static_cast<size_t>(fsize) / rec_bytes;
    d = dim;
}

// 读取所有 fvecs 数据到内存
static void read_fvecs_all(const std::string& path, float* data, size_t n, int d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { std::cerr << "无法打开 fvecs 文件: " << path << std::endl; std::exit(1); }
    for (size_t i = 0; i < n; ++i) {
        int di = 0;
        fin.read(reinterpret_cast<char*>(&di), 4);
        if (!fin || di != d) {
            std::cerr << "记录维度不一致，记录 #" << i << " 期望 " << d << " 实际 " << di << std::endl; std::exit(1);
        }
        fin.read(reinterpret_cast<char*>(data + i * d), sizeof(float) * d);
        if (!fin) { std::cerr << "读取向量失败，记录 #" << i << std::endl; std::exit(1); }
    }
}

// 写 ivecs（每条记录：int dim + dim 个 int）
static void write_ivecs(const std::string& path, const std::vector<int>& flat, int dim_per_record) {
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) { std::cerr << "无法写入 ivecs: " << path << std::endl; std::exit(1); }
    const size_t Q = flat.size() / static_cast<size_t>(dim_per_record);
    for (size_t i = 0; i < Q; ++i) {
        int d = dim_per_record;
        fout.write(reinterpret_cast<const char*>(&d), 4);
        fout.write(reinterpret_cast<const char*>(flat.data() + i * dim_per_record), sizeof(int) * dim_per_record);
    }
}

// 新增：保存超级节点成员 (super2original) 
static void save_supernode_members(const std::string& path,
                                   const std::vector<std::vector<int>>& super_members) {
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout) { std::cerr << "无法写入超级节点成员文件: " << path << std::endl; std::exit(1); }
    int S = (int)super_members.size();
    fout.write(reinterpret_cast<const char*>(&S), 4); // 总超级节点数
    for (int sid = 0; sid < S; ++sid) {
        int cnt = (int)super_members[sid].size();
        fout.write(reinterpret_cast<const char*>(&cnt), 4);
        fout.write(reinterpret_cast<const char*>(super_members[sid].data()), sizeof(int) * cnt);
    }
}


// 随机映射
static void gen_mapping(size_t N_used, int group_size,
                        std::vector<int>& orig2super,
                        std::vector<std::vector<int>>& super_members) {
    if (group_size <= 1) {
        orig2super.resize(N_used);
        super_members.resize(N_used);
        for (size_t i = 0; i < N_used; ++i) {
            orig2super[i] = (int)i;
            super_members[i].push_back((int)i);
        }
        return;
    }

    size_t S = (N_used + group_size - 1) / group_size; // 组数保持不变
    orig2super.resize(N_used);
    super_members.assign(S, {});

    // 打乱原始节点 ID
    std::vector<int> ids;
    ids.reserve(N_used);
    for (size_t i = 0; i < N_used; ++i) ids.push_back((int)i);

    std::mt19937 rng(static_cast<uint32_t>(random_seed));
    std::shuffle(ids.begin(), ids.end(), rng);

    // 按块分配到各组（每组至多 group_size 个，最后一组可能不足）
    for (size_t s = 0; s < S; ++s) {
        size_t start = s * (size_t)group_size;
        size_t end = std::min(start + (size_t)group_size, N_used);
        for (size_t j = start; j < end; ++j) {
            int oid = ids[j];
            orig2super[(size_t)oid] = (int)s;
            super_members[(size_t)s].push_back(oid);
        }
    }
}

// k-hop 过滤器
class KHopFilter : public hnswlib::BaseFilterFunctor {
public:
    explicit KHopFilter(const std::unordered_set<hnswlib::labeltype>& nbrs) : allow_(nbrs) {}
    bool operator()(hnswlib::labeltype id) override {
        return allow_.count(id) > 0;
    }
private:
    const std::unordered_set<hnswlib::labeltype>& allow_;
};

// // 暴力精确 Top-K（带 k-hop 过滤）
// static std::priority_queue<std::pair<float, hnswlib::labeltype>>
// exactKnnWithGraphLimit(const float* query_data, size_t k, hnswlib::labeltype query_label,
//                        hnswlib::BruteforceSearch<float>* bf_index,
//                        hnswlib::GraphRelationSampler* grs, int k_hop) {
//     auto khop_nbr = grs->getKHopNodes(query_label, k_hop);
//     KHopFilter filter(khop_nbr);
//     return bf_index->searchKnn(query_data, k, &filter);
// }

// 使用超级节点 k-hop 限制的精确 Top-K
static std::priority_queue<std::pair<float, hnswlib::labeltype>>
exactKnnWithSupernodeKHop(const float* query_data, size_t k,
                          hnswlib::labeltype query_orig_id,
                          hnswlib::BruteforceSearch<float>* bf_index,
                          hnswlib::GraphRelationSampler* grs,
                          const std::vector<int>& orig2super,
                          const std::vector<std::vector<int>>& super_members,
                          int k_hop) {
    int q_super = orig2super[(size_t)query_orig_id];
    auto khop_supers = grs->getKHopNodes(q_super, k_hop);
    std::unordered_set<hnswlib::labeltype> allow;
    for (auto sid : khop_supers) {
        const auto& vec = super_members[sid];
        for (int oid : vec) allow.insert((hnswlib::labeltype)oid);
    }
    KHopFilter filter(allow);
    return bf_index->searchKnn(query_data, k, &filter);
}

// 新增：分析 k-hop 覆盖率
static void analyzeKHopDistribution(hnswlib::GraphRelationSampler& grs, int max_nodes, int max_hops) {
    std::cout << "\n================ k-hop 覆盖率统计 =================\n";
    std::cout << "Hop  平均邻居数量  占总节点比例(%)\n";
    const int num_samples = std::min(100, max_nodes);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, max_nodes - 1);
    for (int hop = 1; hop <= max_hops; ++hop) {
        long long total = 0;
        for (int i = 0; i < num_samples; ++i) {
            int id = uni(rng);
            auto nbrs = grs.getKHopNodes(id, hop);
            total += (long long)nbrs.size();
        }
        double avg = (double)total / (double)num_samples;
        double pct = (avg / (double)max_nodes) * 100.0;
        std::cout << std::setw(3) << hop
                  << std::setw(12) << (long long)(avg + 0.5)
                  << std::setw(15) << std::fixed << std::setprecision(3) << pct << "\n";
    }
    std::cout << "==================================================\n";
}

// 修改：分析 k-hop 覆盖率（超级节点 & 展开后原始节点）
static void analyzeKHopDistributionSuper(hnswlib::GraphRelationSampler& grs,
                                         const std::vector<std::vector<int>>& super_members,
                                         int max_hops) {
    int S = (int)super_members.size();
    std::cout << "\n============ 超级节点 k-hop 覆盖率统计 ============\n";
    std::cout << "Hop  平均超级节点数  超级节点占比(%)  平均原始节点数  原始占比(%)\n";
    int num_samples = std::min(100, S);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, S - 1);
    for (int hop = 1; hop <= max_hops; ++hop) {
        long long total_super = 0;
        long long total_orig = 0;
        for (int i = 0; i < num_samples; ++i) {
            int sid = uni(rng);
            auto nbrs = grs.getKHopNodes(sid, hop);
            total_super += (long long)nbrs.size();
            long long expand = 0;
            for (auto nsid : nbrs) expand += (long long)super_members[nsid].size();
            total_orig += expand;
        }
        double avg_super = (double)total_super / (double)num_samples;
        double pct_super = (avg_super / (double)S) * 100.0;
        double avg_orig = (double)total_orig / (double)num_samples;
        long long N_orig_total = 0;
        for (auto& v : super_members) N_orig_total += v.size();
        double pct_orig = (avg_orig / (double)N_orig_total) * 100.0;
        std::cout << std::setw(3) << hop
                  << std::setw(14) << (long long)(avg_super + 0.5)
                  << std::setw(15) << std::fixed << std::setprecision(3) << pct_super
                  << std::setw(16) << (long long)(avg_orig + 0.5)
                  << std::setw(12) << std::fixed << std::setprecision(3) << pct_orig
                  << "\n";
    }
    std::cout << "==================================================\n";
}

// 打印使用说明
static void print_usage(const char* prog) {
    std::cout <<
    "用法: " << prog << " [选项]\n"
    "选项:\n"
    "  --fvecs PATH           fvecs 文件路径\n"
    "  --max-elements N       读取的最大向量数 (默认 " << max_elements << ")\n"
    "  --output-dir DIR       输出目录 (默认 " << output_dir << ")\n"
    "  --prob P               概率图建边概率 (默认 " << prob << ")\n"
    "  --topk K               Top-K (默认 " << topk << ")\n"
    "  --k-query Q            查询数量 (默认 " << k_query << ")\n"
    "  --seed S               随机种子 (默认 " << random_seed << ")\n"
    "  --k-hop-min H          hop 下界 (默认 " << k_hop_min << ")\n"
    "  --k-hop-max H          hop 上界 (默认 " << k_hop_max << ")\n"
    "  --map-group-size SIZE  映射分组大小 (默认 " << map_group_size << ")\n"
    "  --graph-model MODEL    图生成模型: er / lfr (默认 " << graph_model << ")\n"
    "  --lfr-avg-degree D     LFR 平均度目标 (默认 " << lfr_avg_degree << ")\n"
    "  --lfr-max-degree D     LFR 最大度 (默认 " << lfr_max_degree << ")\n"
    "  --lfr-degree-tau T     LFR 度分布幂律指数 tau1 (默认 " << lfr_degree_tau << ")\n"
    "  --lfr-community-tau T  LFR 社区大小幂律指数 tau2 (默认 " << lfr_community_tau << ")\n"
    "  --lfr-mu MU            LFR 跨社区边比例 mixing parameter (默认 " << lfr_mu << ")\n"
    "  --lfr-min-community C  LFR 最小社区大小 (默认 " << lfr_min_community_size << ")\n"
    "  --lfr-max-community C  LFR 最大社区大小 (默认 " << lfr_max_community_size << ")\n"
    "  --help                 显示帮助\n";
}

// 解析命令行参数（--key value）
static void parse_args(int argc, char** argv) {
    auto need_val = [&](int i) {
        if (i + 1 >= argc) {
            std::cerr << "缺少参数值: " << argv[i] << std::endl;
            print_usage(argv[0]);
            std::exit(1);
        }
    };
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (key == "--fvecs") {
            need_val(i); fvecs_path = argv[++i];
        } else if (key == "--max-elements") {
            need_val(i); max_elements = std::stoi(argv[++i]);
        } else if (key == "--output-dir") {
            need_val(i); output_dir = argv[++i];
        } else if (key == "--prob") {
            need_val(i); prob = std::stof(argv[++i]);
        } else if (key == "--topk") {
            need_val(i); topk = std::stoi(argv[++i]);
        } else if (key == "--k-query") {
            need_val(i); k_query = std::stoi(argv[++i]);
        } else if (key == "--seed") {
            need_val(i); random_seed = static_cast<uint64_t>(std::stoll(argv[++i]));
        } else if (key == "--k-hop-min") {
            need_val(i); k_hop_min = std::stoi(argv[++i]);
        } else if (key == "--k-hop-max") {
            need_val(i); k_hop_max = std::stoi(argv[++i]);
        } else if (key == "--map-group-size") {
            need_val(i); map_group_size = std::stoi(argv[++i]);
        } else if (key == "--metric") {
            need_val(i); metric = argv[++i];
        } else if (key == "--graph-model") {
            need_val(i); graph_model = argv[++i];
        } else if (key == "--lfr-avg-degree") {
            need_val(i); lfr_avg_degree = std::stoi(argv[++i]);
        } else if (key == "--lfr-max-degree") {
            need_val(i); lfr_max_degree = std::stoi(argv[++i]);
        } else if (key == "--lfr-degree-tau") {
            need_val(i); lfr_degree_tau = std::stod(argv[++i]);
        } else if (key == "--lfr-community-tau") {
            need_val(i); lfr_community_tau = std::stod(argv[++i]);
        } else if (key == "--lfr-mu") {
            need_val(i); lfr_mu = std::stod(argv[++i]);
        } else if (key == "--lfr-min-community") {
            need_val(i); lfr_min_community_size = static_cast<size_t>(std::stoll(argv[++i]));
        } else if (key == "--lfr-max-community") {
            need_val(i); lfr_max_community_size = static_cast<size_t>(std::stoll(argv[++i]));
        } else {
            std::cerr << "未知参数: " << key << std::endl;
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    // 基本校验与修正
    if (max_elements <= 0) { std::cerr << "max-elements 必须为正数\n"; std::exit(1); }
    if (!(prob > 0.0f && prob <= 1.0f)) {
        std::cerr << "prob 必须在 (0, 1]，已修正为 0.0003\n";
        prob = 0.0003f;
    }
    if (topk <= 0) { std::cerr << "topk 必须为正数\n"; std::exit(1); }
    if (k_query <= 0) { std::cerr << "k-query 必须为正数\n"; std::exit(1); }
    if (k_hop_min <= 0 || k_hop_max <= 0) {
        std::cerr << "k-hop 范围必须为正数\n"; std::exit(1);
    }
    if (k_hop_min > k_hop_max) std::swap(k_hop_min, k_hop_max);
    if (map_group_size <= 0) { std::cerr << "map-group-size 必须为正数\n"; std::exit(1); }
    if (!(metric == "l2" || metric == "cosine")) {
        std::cerr << "metric 仅支持 l2 或 cosine\n"; std::exit(1);
    }
    if (!(graph_model == "er" || graph_model == "lfr")) {
        std::cerr << "graph-model 仅支持 er 或 lfr\n"; std::exit(1);
    }
    if (lfr_avg_degree <= 0) { std::cerr << "lfr-avg-degree 必须为正数\n"; std::exit(1); }
    if (lfr_max_degree <= 0) { std::cerr << "lfr-max-degree 必须为正数\n"; std::exit(1); }
    if (lfr_max_degree < lfr_avg_degree) lfr_max_degree = lfr_avg_degree;
    if (lfr_mu < 0.0 || lfr_mu > 1.0) {
        std::cerr << "lfr-mu 必须在 [0, 1]，已修正到 0.3\n";
        lfr_mu = 0.3;
    }
    if (lfr_min_community_size < 2) lfr_min_community_size = 2;
    if (lfr_max_community_size < lfr_min_community_size) lfr_max_community_size = lfr_min_community_size;
}

// 向量归一化（用于 cosine）
static void normalize_vector(float* v, int d) {
    double norm2 = 0.0;
    for (int i = 0; i < d; ++i) norm2 += (double)v[i] * (double)v[i];
    if (norm2 <= 0.0) return;
    float inv = 1.0f / static_cast<float>(std::sqrt(norm2));
    for (int i = 0; i < d; ++i) v[i] *= inv;
}

int main(int argc, char** argv) {
    auto start_time =  std::chrono::high_resolution_clock::now();

    // 解析命令行
    parse_args(argc, argv);

    // // 确保输出目录存在
    // if (!ensure_dir_exists(output_dir)) {
    //     std::cerr << "创建输出目录失败: " << output_dir << std::endl;
    //     return 1;
    // }

    std::cout << "参数汇总:\n"
              << "  fvecs_path=" << fvecs_path << "\n"
              << "  max_elements=" << max_elements << "\n"
              << "  output_dir=" << output_dir << "\n"
              << "  prob=" << prob << "\n"
              << "  topk=" << topk << "\n"
              << "  k_query=" << k_query << "\n"
              << "  random_seed=" << random_seed << "\n"
              << "  k_hop_min=" << k_hop_min << "\n"
              << "  k_hop_max=" << k_hop_max << "\n"
              << "  map_group_size=" << map_group_size << "\n"
              << "  metric=" << metric << "\n"
              << "  graph_model=" << graph_model << "\n";
    if (graph_model == "lfr") {
        std::cout << "  lfr_avg_degree=" << lfr_avg_degree << "\n"
                  << "  lfr_max_degree=" << lfr_max_degree << "\n"
                  << "  lfr_degree_tau=" << lfr_degree_tau << "\n"
                  << "  lfr_community_tau=" << lfr_community_tau << "\n"
                  << "  lfr_mu=" << lfr_mu << "\n"
                  << "  lfr_min_community_size=" << lfr_min_community_size << "\n"
                  << "  lfr_max_community_size=" << lfr_max_community_size << "\n";
    }

    // 1) 读取 base.fvecs
    int dim = 0; size_t N = 0;
    read_fvecs_meta(fvecs_path, dim, N);
    size_t N_total = N;
    size_t N_used = std::min(N_total, static_cast<size_t>(max_elements));
    std::cout << "读取 base: " << fvecs_path << " 维度=" << dim
              << " 总数量=" << N_total << " 实际使用=" << N_used << std::endl;

    std::unique_ptr<float[]> data(new float[N_used * (size_t)dim]);
    auto t0 = std::chrono::high_resolution_clock::now();
    read_fvecs_all(fvecs_path, data.get(), N_used, dim);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "加载完成，用时 " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << " s\n";

    // 如果使用余弦距离，先归一化数据
    if (metric == "cosine") {
        auto t_norm_start = std::chrono::high_resolution_clock::now();
        std::cout << "检测到 metric=cosine，对数据进行归一化...\n";
        for (size_t i = 0; i < N_used; ++i) {
            normalize_vector(data.get() + i * (size_t)dim, dim);
        }
        std::cout << "归一化完成\n";
        auto t_norm_end = std::chrono::high_resolution_clock::now();
        std::cout << "归一化用时 "
              << std::chrono::duration_cast<std::chrono::seconds>(t_norm_end - t_norm_start).count() << " s\n";
    }
    

    // 2.1) 生成映射（超级节点）
    std::vector<int> orig2super;
    std::vector<std::vector<int>> super_members;
    gen_mapping(N_used, map_group_size, orig2super, super_members);
    size_t S = super_members.size();
    std::cout << "超级节点生成: group_size=" << map_group_size
              << " 原始节点=" << N_used << " 超级节点数=" << S << std::endl;

    // 保存映射
    {
        // original_to_super: 维度=1
        std::vector<int> flat;
        flat.reserve(N_used);
        for (size_t i = 0; i < N_used; ++i) flat.push_back(orig2super[i]);
        std::string map_path = output_dir + "/mapping_original_to_supernode.ivecs";
        write_ivecs(map_path, flat, 1);
        std::cout << "映射保存: " << map_path << " (ivecs, dim=1: supernode_id)\n";

        std::string members_path = output_dir + "/supernode_members.bin";
        save_supernode_members(members_path, super_members);
        std::cout << "超级节点成员保存: " << members_path << std::endl;
    }

    // 2) 生成超级节点图
    std::cout << "生成超级节点图: model=" << graph_model;
    if (graph_model == "er") {
        std::cout << ", prob=" << prob;
    } else {
        std::cout << ", avg_degree=" << lfr_avg_degree
                  << ", max_degree=" << lfr_max_degree
                  << ", tau1=" << lfr_degree_tau
                  << ", tau2=" << lfr_community_tau
                  << ", mu=" << lfr_mu
                  << ", community=[" << lfr_min_community_size
                  << "," << lfr_max_community_size << "]";
    }
    std::cout << " ..." << std::endl;
    hnswlib::GraphRelationSampler grs(prob);
    std::vector<size_t> super_ids(S);
    for (size_t i = 0; i < S; ++i) super_ids[i] = i;
    auto tg0 = std::chrono::high_resolution_clock::now();
    if (graph_model == "lfr") {
        grs.genLFRRelation(super_ids.data(),
                           S,
                           lfr_avg_degree,
                           lfr_max_degree,
                           lfr_degree_tau,
                           lfr_community_tau,
                           lfr_mu,
                           lfr_min_community_size,
                           lfr_max_community_size,
                           random_seed);
    } else {
        grs.genRelation(super_ids.data(), S);
    }
    auto tg1 = std::chrono::high_resolution_clock::now();
    std::cout << "超级节点图生成完成，用时 "
              << std::chrono::duration_cast<std::chrono::seconds>(tg1 - tg0).count() << " s\n";
    grs.printInfo();

    // 覆盖率分析（超级节点）
    analyzeKHopDistributionSuper(grs, super_members, k_hop_max);

    // 保存超级节点图
    grs.saveRelation(output_dir + "/graph.bin");

    // 3) 构建原始向量暴力索引
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    if (metric == "cosine") {
        space.reset(new hnswlib::InnerProductSpace(dim));
    } else {
        space.reset(new hnswlib::L2Space(dim));
    }
    hnswlib::BruteforceSearch<float> bf_index(space.get(), N_used);
    std::cout << "构建 BruteForce 索引并插入向量 ..." << std::endl;
    auto tb0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N_used; ++i) bf_index.addPoint(data.get() + i * dim, i);
    auto tb1 = std::chrono::high_resolution_clock::now();
    std::cout << "BF 索引构建完成，用时 " << std::chrono::duration_cast<std::chrono::seconds>(tb1 - tb0).count() << " s\n";

    // 4) 采样查询 (id, k_hop)
    const int Q = std::min<int>(k_query, static_cast<int>(N_used));
    std::vector<int> query_pairs;           // 扁平化存储：[id0, hop0, id1, hop1, ...]
    query_pairs.resize(Q * 2);

    std::mt19937 rng(static_cast<uint32_t>(random_seed));
    // std::uniform_int_distribution<int> hop_dist(k_hop_min, k_hop_max);

    for (int i = 0; i < Q; ++i) {
        int qid = static_cast<int>(rng() % N_used);
        // int hop = hop_dist(rng);
        int hop = k_hop_min; // 固定 hop
        query_pairs[i * 2 + 0] = qid;
        query_pairs[i * 2 + 1] = hop;
    }

    // 保存查询序列 (ivecs, dim=2: [id, k_hop])
    const std::string queries_path = output_dir + "/queries.ivecs";
    write_ivecs(queries_path, query_pairs, 2);
    std::cout << "查询序列已保存: " << queries_path << " (ivecs, dim=2: [id, k_hop])\n";

    // 5) 生成 groundtruth（仅 labels）
    // TODO: 并行生成 gt
    std::vector<int> gt_labels; gt_labels.resize((size_t)Q * topk, -1);

    std::cout << "开始暴力检索生成 groundtruth（仅 labels，按各自 k-hop 过滤）: "
              << "K=" << topk << std::endl;

    auto tq0 = std::chrono::high_resolution_clock::now();
    long long sum_us = 0; // 统计每次查询（exactKnnWithGraphLimit）的总耗时，微秒
    for (int qi = 0; qi < Q; ++qi) {
        int qid = query_pairs[qi * 2 + 0];
        int k_hop_i = query_pairs[qi * 2 + 1];

        const float* qvec = data.get() + (size_t)qid * dim;

        // 计时（包含 k-hop 邻域生成与 BruteForce 搜索）
        auto tqi1 = std::chrono::high_resolution_clock::now();
        auto res = exactKnnWithSupernodeKHop(qvec, (size_t)topk,
                                             (hnswlib::labeltype)qid,
                                             &bf_index, &grs,
                                             orig2super, super_members,
                                             k_hop_i);
        auto tqi2 = std::chrono::high_resolution_clock::now();
        sum_us += std::chrono::duration_cast<std::chrono::microseconds>(tqi2 - tqi1).count();

        // 注意：优先队列为最大堆，这里从末尾向前填，确保结果按距离升序
        int base = qi * topk;
        int cnt = 0;
        while (!res.empty() && cnt < topk) {
            gt_labels[base + (topk - 1 - cnt)] = static_cast<int>(res.top().second);
            res.pop();
            cnt++;
        }

        if ((qi + 1) % 1000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration_cast<std::chrono::seconds>(now - tq0).count();
            std::cout << "\r进度: " << (qi + 1) << "/" << Q << " 用时 " << (int)sec << " s" << std::flush;
        }
    }
    auto tq1 = std::chrono::high_resolution_clock::now();
    std::cout << "\n检索完成，用时 " << std::chrono::duration_cast<std::chrono::seconds>(tq1 - tq0).count() << " s\n";

    // 统计结果：平均耗时(us)与 QPS，并保存到文件
    double avg_us = Q > 0 ? (double)sum_us / (double)Q : 0.0;
    double qps = (avg_us > 0.0) ? (1e6 / avg_us) : 0.0;
    long long wall_us = std::chrono::duration_cast<std::chrono::microseconds>(tq1 - tq0).count();

    std::cout << "BruteForce+kHop 平均耗时: " << (long long)avg_us << " us/query, QPS: " << qps << std::endl;

    const std::string bf_stats_path = output_dir + "/bf_search_stats.txt";
    {
        std::ofstream stats(bf_stats_path, std::ios::out | std::ios::trunc);
        if (!stats) {
            std::cerr << "无法写入统计文件: " << bf_stats_path << std::endl;
        } else {
            stats << "method=BruteforceWithKHopFilter\n";
            stats << "Q=" << Q << "\n";
            stats << "K=" << topk << "\n";
            stats << "avg_time_us=" << avg_us << "\n";
            stats << "qps=" << qps << "\n";
            stats << "sum_time_us=" << sum_us << "\n";
            stats << "wall_time_us=" << wall_us << "\n";
            stats << "note=avg_time_us 基于每次 exactKnnWithGraphLimit 的耗时（包含 k-hop 邻域生成与 BF 搜索）\n";
        }
    }
    std::cout << "统计结果已保存: " << bf_stats_path << std::endl;

    // 6) 写出 groundtruth（仅 labels）
    const std::string gt_lbl_path = output_dir + "/groundtruth.ivecs";
    write_ivecs(gt_lbl_path, gt_labels, topk);
    std::cout << "groundtruth 已保存: " << gt_lbl_path << std::endl;

    // 7) 写出参数记录增加映射 & 超级节点信息
    {
        const std::string meta_path = output_dir + "/meta.txt";
        std::ofstream meta(meta_path, std::ios::out | std::ios::trunc);
        if (!meta) {
            std::cerr << "无法写入参数文件: " << meta_path << std::endl;
        } else {
            meta << "fvecs_path=" << fvecs_path << "\n";
            meta << "output_dir=" << output_dir << "\n";
            meta << "dim=" << dim << "\n";
            meta << "N_total=" << N_total << "\n";
            meta << "N_used=" << N_used << "\n";
            meta << "prob=" << prob << "\n";
            meta << "graph_model=" << graph_model << "\n";
            meta << "lfr_avg_degree=" << lfr_avg_degree << "\n";
            meta << "lfr_max_degree=" << lfr_max_degree << "\n";
            meta << "lfr_degree_tau=" << lfr_degree_tau << "\n";
            meta << "lfr_community_tau=" << lfr_community_tau << "\n";
            meta << "lfr_mu=" << lfr_mu << "\n";
            meta << "lfr_min_community_size=" << lfr_min_community_size << "\n";
            meta << "lfr_max_community_size=" << lfr_max_community_size << "\n";
            meta << "topk=" << topk << "\n";
            meta << "k_query=" << k_query << "\n";
            // 记录 hop 的范围与 queries 文件格式
            meta << "k_hop_min=" << k_hop_min << "\n";
            meta << "k_hop_max=" << k_hop_max << "\n";
            meta << "map_group_size=" << map_group_size << "\n";
            meta << "metric=" << metric << "\n";
            meta << "supernodes=" << S << "\n";
            meta << "mapping_original_to_supernode=" << (output_dir + "/mapping_original_to_supernode.ivecs") << "\n";
            meta << "supernode_members_file=" << (output_dir + "/supernode_members.bin") << "\n";
            // meta << "graph_supernodes_file=" << (output_dir + "/graph_supernodes.bin") << "\n";
            meta << "queries_file=" << (output_dir + "/queries.ivecs") << " (ivecs, dim=2: [id, k_hop])\n";
            meta << "random_seed=" << random_seed << "\n";
            meta << "graph_file=" << (output_dir + "/graph.bin") << "\n";
            meta << "groundtruth_labels=" << gt_lbl_path << "\n";
            meta << "bf_search_stats=" << bf_stats_path << "\n";
        }
        std::cout << "参数文件已更新(含映射): " << meta_path << std::endl;
    }

    auto end_time =  std::chrono::high_resolution_clock::now();
    std::cout << "全部完成，总用时 " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << " s\n";

    return 0;
}
