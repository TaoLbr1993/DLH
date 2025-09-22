#pragma once
#include "hnswlib.h"
#include <unordered_set>
#include <random>
#include <vector>
#include <unordered_set>

namespace hnswlib {

    // compare struct and operator: used in searchKnn
    template<typename dist_t>
    class HopNbrTrible {
    public:
        bool in_hop = false;
        dist_t dist;
        tableint id;
        HopNbrTrible(bool in_hop_, dist_t dist_, tableint id_):in_hop(in_hop_), dist(dist_), id(id_){}
    };

    template<typename dist_t>
    bool operator<(const HopNbrTrible<dist_t> &a, const HopNbrTrible<dist_t> &b) {
        // std::cout << (int)(!(a.in_hop)) << " " << (int)(!(b.in_hop)) << std::endl;
        // return (int)(!(a.in_hop))*0.05+a.dist < (int)(!(b.in_hop))*0.1+b.dist;
        return a.dist < b.dist;
        if (a.in_hop && !b.in_hop) return false;
        if (!a.in_hop && b.in_hop) return true;
        return a.dist < b.dist;
    }

    template<typename dist_t>
    class HopNbrTribleV2 {
        public:
            int hop_range;
            int hop;
            dist_t dist;
            tableint id;

        HopNbrTribleV2(int hop_range_, int hop_, dist_t dist_, tableint id_):hop_range(hop_range_), hop(hop_), dist(dist_), id(id_){}
    };

    template<typename dist_t>
    bool operator<(const HopNbrTribleV2<dist_t> &a, const HopNbrTribleV2<dist_t> &b) {
        // std::cout << (int)(!(a.in_hop)) << " " << (int)(!(b.in_hop)) << std::endl;
        // return (int)(!(a.in_hop))*0.05+a.dist < (int)(!(b.in_hop))*0.1+b.dist;
        // return a.dist < b.dist;

        if ((a.hop >= -a.hop_range && b.hop >= -b.hop_range) || (a.hop < a.hop_range && b.hop < b.hop_range)) return a.dist < b.dist;
        return (a.hop < b.hop);
        if (a.hop < b.hop) return true;
        if (a.hop > b.hop) return false;
        return a.dist < b.dist;
    }

    //todo: generate a interface
    class GraphRelationSampler {
    public:
        float prob;
        std::unordered_map<labeltype, size_t> id_start_point_map;
        std::unordered_map<labeltype, unsigned int> offset_map;
        size_t * end_points{nullptr};
        // endpoints of id1 + endpoints of id2 + ...

        GraphRelationSampler(float prob):prob(prob){}

        void clear() {
            // 清空原有数据
            id_start_point_map.clear();
            offset_map.clear();
            if (end_points) {
                delete[] end_points;
                end_points = nullptr;
            }
        }

        void genRelation(size_t* ids, size_t num_ids) {
            // todo:
            // generate graph information upon ids
            // distribution:
            // https://www.cnblogs.com/orion-orion/p/16254923.html 
            // Gnp
            
            // 清空原有数据
            clear();

            // 邻接表暂存一下
            std::vector<std::vector<size_t>> edges(num_ids);

            // 生成随机种子并使用种子初始化随机数生成器
            std::random_device rd;
            std::default_random_engine rng(rd());
            // 使用均匀分布生成随机数
            std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

            // 遍历每对节点（无向图，不重复）
            for (size_t i = 0; i < num_ids; i++) {
                for (size_t j = i + 1; j < num_ids; j++) {
                    float s = distrib(rng);
                    if (s < prob) {
                        // 添加无向边
                        edges[i].push_back(ids[j]);
                        edges[j].push_back(ids[i]);
                    }
                }
            }

            // 统计总边数
            size_t totalEdges = 0;
            for (size_t i = 0; i < num_ids; ++i) {
                totalEdges += edges[i].size();
            }

            // 分配 end_points
            end_points = new size_t[totalEdges];

            // 填充 CSR 结构
            size_t pos = 0;
            for (size_t i = 0; i < num_ids; i++) {
                labeltype id = ids[i];
                id_start_point_map[id] = pos;
                offset_map[id] = static_cast<unsigned int>(edges[i].size());
                for (size_t v : edges[i]) {
                    end_points[pos++] = v;
                }
            }
        }
    
        void saveRelation(const std::string & location) {
            // todo:
            // save sampled graph information
            // refer to:
            // hnswalg.h/HerarchicalNSW->saveIndex
            std::ofstream output(location, std::ios::binary);
            if (!output.is_open()) {
                throw std::runtime_error("Cannot open file for saving relation: " + location);
            }

            // 保存 prob
            writeBinaryPOD(output, prob);

            // 保存 id_start_point_map 大小和内容
            size_t idStartPointMapSize = id_start_point_map.size();
            writeBinaryPOD(output, idStartPointMapSize);

            for (const auto& kv : id_start_point_map) {
                writeBinaryPOD(output, kv.first);
                writeBinaryPOD(output, kv.second);
            }

            // 保存 offset_map 大小和内容
            size_t offsetMapSize = offset_map.size();
            writeBinaryPOD(output, offsetMapSize);
            for (const auto& kv : offset_map) {
                writeBinaryPOD(output, kv.first);
                writeBinaryPOD(output, kv.second);
            }

            // 保存 end_points 长度和内容
            size_t totalEdges = 0;
            for (const auto& kv : offset_map) {
                totalEdges += kv.second;
            }
            writeBinaryPOD(output, totalEdges);
            output.write((char*)end_points, sizeof(size_t) * totalEdges);

            output.close();
        }

        void loadRelation(const std::string & location) {
            // todo:
            // load sampled graph information
            // refer to:
            // hnswalg.h/HerarchicalNSW->loadIndex 
            std::ifstream input(location, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("Cannot open file for loading relation: " + location);
            }

            // 清空原有数据
            clear();

            // 读取 prob
            readBinaryPOD(input, prob);

            // 读取 id_start_point_map
            size_t idStartPointMapSize = 0;
            readBinaryPOD(input, idStartPointMapSize);
            for (size_t i = 0; i < idStartPointMapSize; i++) {
                labeltype key;
                size_t value;
                readBinaryPOD(input, key);
                readBinaryPOD(input, value);
                id_start_point_map[key] = value;
            }

            // 读取 offset_map
            size_t offsetMapSize = 0;
            readBinaryPOD(input, offsetMapSize);
            for (size_t i = 0; i < offsetMapSize; i++) {
                labeltype key;
                unsigned int value;
                readBinaryPOD(input, key);
                readBinaryPOD(input, value);
                offset_map[key] = value;
            }

            // 读取 end_points
            size_t totalEdges = 0;
            readBinaryPOD(input, totalEdges);
            if (totalEdges > 0) {
                end_points = new size_t[totalEdges];
                input.read(reinterpret_cast<char*>(end_points), sizeof(size_t) * totalEdges);
            } else {
                end_points = nullptr;
            }

            input.close();
        }

        // 根据给定的 id 和 k 值，获取 k-hop 节点集合（最多经过 k 条边到达的节点，不包括自身）
        std::unordered_set<labeltype> getKHopNodes(labeltype id, int k) {
            auto it_start = id_start_point_map.find(0);
            std::unordered_set<labeltype> result;
            std::unordered_set<labeltype> visited;
            std::unordered_set<labeltype> current_level;
            current_level.insert(id);
            visited.insert(id);
            for (int hop = 0; hop < k; hop++) {
                std::unordered_set<labeltype> next_level;

                for (const auto& node : current_level) {
                    auto it_start = id_start_point_map.find(node);
                    auto it_offset = offset_map.find(node);
                    if (it_start == id_start_point_map.end() || it_offset == offset_map.end()) continue;
                    size_t start = it_start->second;
                    unsigned int offset = it_offset->second;
                    for (size_t i = 0; i < offset; i++) {
                        labeltype neighbor = end_points[start + i];
                        if (visited.find(neighbor) == visited.end()) {
                            next_level.insert(neighbor);
                            visited.insert(neighbor);
                        }
                    }
                }
                if (next_level.empty()) break;
                result.insert(next_level.begin(), next_level.end());
                current_level = std::move(next_level);
            }
            return result;
        }

        std::unordered_map<labeltype, int> getKHopNodesWiDist(labeltype id, int k) {
            std::unordered_map<labeltype, int> result;
            std::unordered_set<labeltype> visited;
            std::unordered_set<labeltype> current_level;

            current_level.insert(id);
            visited.insert(id);

            for (int hop=0; hop<k; hop++) {
                std::unordered_set<labeltype> next_level;
                for (const auto& node: current_level) {

                    auto it_start = id_start_point_map.find(node);
                    auto it_offset = offset_map.find(node);
                    if (it_start == id_start_point_map.end() || it_offset == offset_map.end()) continue;
                    
                    size_t start = it_start->second;
                    unsigned int offset = it_offset->second;
                    for (size_t i=0; i < offset; i++) {
                        labeltype neighbor = end_points[start+i];
                        if (visited.find(neighbor) == visited.end()) {
                            next_level.insert(neighbor);
                            visited.insert(neighbor);
                            result.emplace(neighbor, hop+1);
                        }
                    }
                }
                if (next_level.empty()) break;
                current_level = std::move(next_level);
            }
            return result;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<int, labeltype> const& a,
                std::pair<int, labeltype> const& b) const noexcept {
                return a.first < b.first;
            }
        };
        std::unordered_set<labeltype> getMaxDegreeNodes(int k) {
            std::priority_queue<std::pair<int, labeltype>, std::vector<std::pair<int, labeltype>>, CompareByFirst> node_pq;
            for (auto kv: offset_map) {
                node_pq.emplace(kv.second, kv.first);
                if (node_pq.size() > k) {
                    node_pq.pop();
                }
            }
            std::unordered_set<labeltype> result;
            while (node_pq.size() > 0){
                result.insert(node_pq.top().second);
                node_pq.pop();
            }
            return result;
        }
    };
}