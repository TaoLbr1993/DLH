#pragma once
#include <cmath>
#include <vector>
#include <cstddef>
#include <unordered_set>
#include <cstdint>


// 对单个向量做 L2 归一化；norm=0 时保持不变
inline void l2_normalize_inplace(float* v, int dim) {
    double norm2 = 0.0;
    for (int i = 0; i < dim; ++i) norm2 += (double)v[i] * (double)v[i];
    if (norm2 <= 0.0) return;
    float inv = 1.0f / static_cast<float>(std::sqrt(norm2));
    for (int i = 0; i < dim; ++i) v[i] *= inv;
}

// 对整批向量做 L2 归一化；data.size() 必须是 dim 的整数倍
inline void l2_normalize_dataset_inplace(std::vector<float>& data, int dim) {
    if (dim <= 0) return;
    const size_t n = data.size() / (size_t)dim;
    for (size_t i = 0; i < n; ++i) {
        l2_normalize_inplace(data.data() + i * (size_t)dim, dim);
    }
}

inline double recall_at_k_from_sets(const std::vector<int>& approx, const std::unordered_set<int>& gt_set) {
    int hit=0;
    for (int v : approx) if (v>=0 && gt_set.count(v)) hit++;
    int K = (int)approx.size();
    if (K==0) return 0.0;
    return (double)hit / (double)K;
}

struct Stats {
    long long sum_us = 0;
    int avg_us = 0;
    double avg_recall = 0.0; // 0..1
};