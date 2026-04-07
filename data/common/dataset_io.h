#pragma once
#include "../../hnswlib/hnswlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>

struct MetaInfo {
    std::string fvecs_path;
    std::string output_dir;
    int dim = 0;
    size_t N_total = 0;
    size_t N_used = 0;
    float prob = 0.0f;
    int topk = 0;
    int k_query = 0;
    int k_hop_min = 1;
    int k_hop_max = 1;
    uint64_t random_seed = 0;
    std::string queries_file;
    std::string graph_file;
    std::string gt_labels_file;
    std::string bf_stats_file; // 可选
    // 新增
    int map_group_size = 1;
    int supernodes = 0;
    std::string mapping_original_to_supernode_file;
    std::string supernode_members_file;
    // 
    std::string metric;
};

struct Queries {
    // 每条记录: [id, k_hop]
    std::vector<int> flat; // Q * 2
    int Q = 0;
};

// 读取 meta 并装载全部必要数据的便捷函数
struct LoadedDataset {
    MetaInfo meta;
    std::vector<float> base; // size = N_used * dim
    Queries queries;
    std::vector<int> gt_labels; // size = Q * K
    int K = 0;
    // 新增
    std::vector<int> orig2super;               // size = N_used
    std::vector<std::vector<int>> super_members; // size = supernodes
};

static void read_fvecs_meta(const std::string& path, int& d, size_t& n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "无法打开 fvecs: " << path << std::endl; std::exit(1); }
    int dim=0; f.read((char*)&dim, 4);
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    size_t rec = 4 + (size_t)dim * 4;
    if (rec==0 || bytes % rec != 0) { std::cerr << "fvecs 尺寸不匹配: " << path << std::endl; std::exit(1); }
    d = dim; n = bytes / rec;
}
static void read_fvecs_all(const std::string& path, float* data, size_t n, int d) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "无法打开 fvecs: " << path << std::endl; std::exit(1); }
    for (size_t i=0;i<n;++i) {
        int di=0; f.read((char*)&di,4);
        if (!f || di!=d) { std::cerr << "记录维度不匹配 #" << i << std::endl; std::exit(1); }
        f.read((char*)(data + i*d), sizeof(float)*d);
    }
}
static void read_ivecs_all(const std::string& path, std::vector<int>& flat, int& drec) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "无法打开 ivecs: " << path << std::endl; std::exit(1); }
    flat.clear();
    std::vector<int> buf;
    while (true) {
        int d=0; f.read((char*)&d, 4);
        if (!f) break;
        if (flat.empty()) drec = d;
        if (d != drec) { std::cerr << "ivecs 记录维度不一致: " << path << std::endl; std::exit(1); }
        buf.resize(d);
        f.read((char*)buf.data(), sizeof(int)*d);
        if (!f) { std::cerr << "读取 ivecs 失败: " << path << std::endl; std::exit(1); }
        flat.insert(flat.end(), buf.begin(), buf.end());
    }
}

static void read_supernode_members(const std::string& path,
                                   std::vector<std::vector<int>>& super_members,
                                   int expected_S) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "无法打开超级节点成员文件: " << path << std::endl; std::exit(1); }
    int S = 0;
    f.read((char*)&S, 4);
    if (!f) { std::cerr << "读取超级节点数失败: " << path << std::endl; std::exit(1); }
    if (expected_S > 0 && S != expected_S) {
        std::cerr << "meta supernodes=" << expected_S << " 但文件中 S=" << S << std::endl;
        std::exit(1);
    }
    super_members.clear();
    super_members.resize(S);
    for (int sid = 0; sid < S; ++sid) {
        int cnt = 0;
        f.read((char*)&cnt, 4);
        if (!f) { std::cerr << "读取成员数失败 sid=" << sid << std::endl; std::exit(1); }
        if (cnt < 0) { std::cerr << "成员数非法 sid=" << sid << std::endl; std::exit(1); }
        super_members[sid].resize(cnt);
        if (cnt > 0) {
            f.read((char*)super_members[sid].data(), sizeof(int) * cnt);
            if (!f) { std::cerr << "读取成员列表失败 sid=" << sid << std::endl; std::exit(1); }
        }
    }
}

static void read_mapping_orig2super(const std::string& path,
                                    std::vector<int>& orig2super,
                                    size_t expected_N) {
    std::vector<int> flat;
    int drec = 0;
    read_ivecs_all(path, flat, drec);
    if (drec != 1) {
        std::cerr << "mapping_original_to_supernode.ivecs 维度应为1, 实际: " << drec << std::endl;
        std::exit(1);
    }
    if (expected_N > 0 && flat.size() != expected_N) {
        std::cerr << "映射记录数与 N_used 不符: map=" << flat.size()
                  << " N_used=" << expected_N << std::endl;
        std::exit(1);
    }
    orig2super = std::move(flat);
}

bool load_meta(const std::string& meta_path, MetaInfo& mi) {
    std::ifstream f(meta_path);
    if (!f) return false;
    std::unordered_map<std::string,std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        auto k = line.substr(0,pos);
        auto v = line.substr(pos+1);
        kv[k]=v;
    }
    auto get = [&](const std::string& k)->std::string {
        return kv.count(k)? kv[k] : std::string();
    };
    mi.fvecs_path = get("fvecs_path");
    mi.output_dir = get("output_dir");
    mi.dim = std::stoi(get("dim"));
    mi.N_total = (size_t)std::stoll(get("N_total"));
    mi.N_used = (size_t)std::stoll(get("N_used"));
    mi.prob = std::stof(get("prob"));
    mi.topk = std::stoi(get("topk"));
    mi.k_query = std::stoi(get("k_query"));
    mi.k_hop_min = std::stoi(get("k_hop_min"));
    mi.k_hop_max = std::stoi(get("k_hop_max"));
    mi.random_seed = (uint64_t)std::stoull(get("random_seed"));
    mi.map_group_size = get("map_group_size").empty()? 1 : std::stoi(get("map_group_size"));
    mi.supernodes = get("supernodes").empty()? 0 : std::stoi(get("supernodes"));
    mi.mapping_original_to_supernode_file = get("mapping_original_to_supernode");
    mi.supernode_members_file = get("supernode_members_file");
    mi.queries_file = get("queries_file");
    auto qpos = mi.queries_file.find(' ');
    if (qpos != std::string::npos) mi.queries_file = mi.queries_file.substr(0, qpos);
    mi.graph_file = get("graph_file");
    mi.gt_labels_file = get("groundtruth_labels");
    mi.bf_stats_file = get("bf_search_stats");

    mi.metric = get("metric");
    
    return true;
}

void load_base_fvecs_all(const std::string& p, float* dst, size_t n, int d) { read_fvecs_all(p, dst, n, d); }
void load_ivecs(const std::string& p, std::vector<int>& out_flat, int& drec) { read_ivecs_all(p, out_flat, drec); }

void load_groundtruth_labels(const std::string& path, std::vector<int>& out_flat, int& K, int& Q) {
    int drec = 0;
    read_ivecs_all(path, out_flat, drec);
    K = drec;
    Q = (int)(out_flat.size() / (size_t)K);
}

Queries load_queries_ivecs2(const std::string& path) {
    Queries q;
    int d=0;
    read_ivecs_all(path, q.flat, d);
    if (d != 2) { std::cerr << "queries.ivecs dim 应为 2, 实际: " << d << std::endl; std::exit(1); }
    q.Q = (int)(q.flat.size()/2);
    return q;
}

bool load_graph(hnswlib::GraphRelationSampler& grs, const std::string& graph_path) {
    try {
        grs.loadRelation(graph_path);
        return true;
    } catch (...) {
        std::cerr << "加载图失败: " << graph_path << std::endl;
        return false;
    }
}

LoadedDataset load_dataset_all(const std::string& data_dir) {
    LoadedDataset ds;
    // meta
    if (!load_meta(data_dir + "/meta.txt", ds.meta)) {
        std::cerr << "无法读取 meta.txt 于: " << data_dir << std::endl; std::exit(1);
    }
    // base
    ds.base.resize(ds.meta.N_used * (size_t)ds.meta.dim);
    load_base_fvecs_all(ds.meta.fvecs_path, ds.base.data(), ds.meta.N_used, ds.meta.dim);
    // queries
    ds.queries = load_queries_ivecs2(data_dir + "/queries.ivecs");
    // groundtruth labels
    int Q=0; load_groundtruth_labels(data_dir + "/groundtruth.ivecs", ds.gt_labels, ds.K, Q);
    if (Q != ds.queries.Q) {
        std::cerr << "groundtruth 与 queries 数量不一致: gtQ="<<Q<<" qQ="<<ds.queries.Q<<std::endl;
        std::exit(1);
    }
    // 读取映射
    if (!ds.meta.mapping_original_to_supernode_file.empty()) {
        read_mapping_orig2super(ds.meta.mapping_original_to_supernode_file,
                                ds.orig2super, ds.meta.N_used);
    }
    // 读取超级节点成员
    if (!ds.meta.supernode_members_file.empty()) {
        read_supernode_members(ds.meta.supernode_members_file,
                               ds.super_members,
                               ds.meta.supernodes);
        if (ds.meta.supernodes > 0 && (int)ds.super_members.size() != ds.meta.supernodes) {
            std::cerr << "supernode_members 与 meta.supernodes 不一致" << std::endl;
            std::exit(1);
        }
    }
    return ds;
}