#pragma once
#include "hnswlib.h"
#include <unordered_set>
#include <random>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstdint>

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
        std::vector<std::pair<int, int>> edge_pairs;
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
            edge_pairs.clear();
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

            // 随机数生成器，固定种子以保证可复现
            std::default_random_engine rng(42);
            // 使用均匀分布生成随机数
            std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

            // 遍历每对节点（无向图，不重复）
            for (size_t i = 0; i < num_ids; i++) {
                for (size_t j = i + 1; j < num_ids; j++) {
                    float s = distrib(rng);
                    if (s < prob) {
                        edge_pairs.push_back(std::make_pair((int)ids[i], (int)ids[j]));
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

        void genLFRRelation(size_t* ids,
                            size_t num_ids,
                            int avg_degree,
                            int max_degree,
                            double degree_tau,
                            double community_tau,
                            double mixing_mu,
                            size_t min_community_size,
                            size_t max_community_size,
                            uint64_t seed) {
            clear();
            if (num_ids == 0) return;
            if (num_ids == 1) {
                std::vector<std::vector<size_t>> edges(1);
                buildFromAdjacency(ids, edges);
                return;
            }

            avg_degree = std::max(1, avg_degree);
            max_degree = std::max(avg_degree, max_degree);
            max_degree = std::min<int>(max_degree, static_cast<int>(num_ids) - 1);
            degree_tau = degree_tau > 0.0 ? degree_tau : 2.5;
            community_tau = community_tau > 0.0 ? community_tau : 1.5;
            mixing_mu = std::max(0.0, std::min(1.0, mixing_mu));
            min_community_size = std::max<size_t>(2, min_community_size);
            max_community_size = std::max(min_community_size, max_community_size);
            max_community_size = std::min(max_community_size, num_ids);

            std::mt19937 rng(static_cast<uint32_t>(seed));
            std::vector<size_t> order(num_ids);
            for (size_t i = 0; i < num_ids; ++i) order[i] = i;
            std::shuffle(order.begin(), order.end(), rng);

            std::vector<size_t> community_sizes;
            size_t remaining = num_ids;
            while (remaining > 0) {
                if (remaining <= max_community_size) {
                    if (remaining < min_community_size && !community_sizes.empty()) {
                        community_sizes.back() += remaining;
                    } else {
                        community_sizes.push_back(remaining);
                    }
                    break;
                }
                size_t sampled = samplePowerLawInt(min_community_size, max_community_size, community_tau, rng);
                sampled = std::min(sampled, remaining);
                if (remaining - sampled > 0 && remaining - sampled < min_community_size) {
                    sampled = remaining;
                }
                community_sizes.push_back(sampled);
                remaining -= sampled;
            }

            std::vector<int> node_community(num_ids, -1);
            std::vector<std::vector<size_t>> communities;
            communities.reserve(community_sizes.size());
            size_t cursor = 0;
            for (size_t cid = 0; cid < community_sizes.size(); ++cid) {
                communities.push_back(std::vector<size_t>());
                communities.back().reserve(community_sizes[cid]);
                for (size_t j = 0; j < community_sizes[cid] && cursor < order.size(); ++j) {
                    size_t node = order[cursor++];
                    node_community[node] = static_cast<int>(cid);
                    communities.back().push_back(node);
                }
            }

            std::vector<double> raw_degree(num_ids);
            double raw_sum = 0.0;
            for (size_t i = 0; i < num_ids; ++i) {
                raw_degree[i] = static_cast<double>(samplePowerLawInt(1, static_cast<size_t>(max_degree), degree_tau, rng));
                raw_sum += raw_degree[i];
            }
            double scale = raw_sum > 0.0 ? (static_cast<double>(avg_degree) * num_ids / raw_sum) : 1.0;

            std::vector<int> degree(num_ids, 1);
            std::vector<int> internal_target(num_ids, 0);
            std::vector<int> external_target(num_ids, 0);
            for (size_t i = 0; i < num_ids; ++i) {
                int cid = node_community[i];
                int community_cap = cid >= 0 ? static_cast<int>(communities[cid].size()) - 1 : 0;
                int deg = static_cast<int>(std::round(raw_degree[i] * scale));
                deg = std::max(1, std::min(max_degree, deg));
                deg = std::min<int>(deg, static_cast<int>(num_ids) - 1);
                int internal = static_cast<int>(std::round((1.0 - mixing_mu) * deg));
                internal = std::max(0, std::min(internal, community_cap));
                degree[i] = deg;
                internal_target[i] = internal;
                external_target[i] = std::max(0, deg - internal);
            }

            std::vector<std::vector<size_t>> edges(num_ids);
            std::vector<int> internal_count(num_ids, 0);

            for (size_t cid = 0; cid < communities.size(); ++cid) {
                const std::vector<size_t>& members = communities[cid];
                if (members.size() < 2) continue;
                std::uniform_int_distribution<size_t> pick(0, members.size() - 1);
                for (size_t idx = 0; idx < members.size(); ++idx) {
                    size_t u = members[idx];
                    size_t attempts = 0;
                    size_t max_attempts = std::max<size_t>(100, members.size() * 10);
                    while (internal_count[u] < internal_target[u] && attempts++ < max_attempts) {
                        size_t v = members[pick(rng)];
                        if (u == v || hasEdge(edges, u, v)) continue;
                        addUndirectedEdge(edges, u, v);
                        internal_count[u]++;
                        internal_count[v]++;
                    }
                }
            }

            for (size_t i = 0; i < num_ids; ++i) {
                int missing_internal = std::max(0, internal_target[i] - internal_count[i]);
                external_target[i] += missing_internal;
            }

            std::vector<int> external_count(num_ids, 0);
            std::uniform_int_distribution<size_t> pick_node(0, num_ids - 1);
            for (size_t u = 0; u < num_ids; ++u) {
                size_t attempts = 0;
                size_t max_attempts = std::max<size_t>(1000, num_ids * 2);
                while (external_count[u] < external_target[u] && attempts++ < max_attempts) {
                    size_t v = pick_node(rng);
                    if (u == v || node_community[u] == node_community[v] || hasEdge(edges, u, v)) continue;
                    addUndirectedEdge(edges, u, v);
                    external_count[u]++;
                    external_count[v]++;
                }
            }

            for (size_t i = 0; i < num_ids; ++i) {
                std::sort(edges[i].begin(), edges[i].end());
            }
            buildFromAdjacency(ids, edges);
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

            // 保存 edge_pairs 大小和内容
            size_t edgePairsSize = edge_pairs.size();
            writeBinaryPOD(output, edgePairsSize);
            for (const auto& p : edge_pairs) {
                writeBinaryPOD(output, p.first);
                writeBinaryPOD(output, p.second);
            }

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

            // 读取 edge_pairs
            size_t edgePairsSize = 0;
            readBinaryPOD(input, edgePairsSize);
            edge_pairs.clear();
            edge_pairs.reserve(edgePairsSize);
            for (size_t i = 0; i < edgePairsSize; ++i) {
                int u, v;
                readBinaryPOD(input, u);
                readBinaryPOD(input, v);
                edge_pairs.emplace_back(u, v);
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
    
        void printInfo() {
            std::cout << "GraphRelationSampler info: " << std::endl;
            std::cout << "  prob: " << prob << std::endl;

            // 基础统计
            size_t nodes_with_edges = id_start_point_map.size();
            size_t totalAdjEntries = 0; // CSR 中的邻接条目数（度数之和）
            for (const auto& kv : offset_map) {
                totalAdjEntries += kv.second;
            }
            size_t undirectedEdges = edge_pairs.size(); // 生成时存了一次无向边(i,j)

            std::cout << "  number of nodes with edges: " << nodes_with_edges << std::endl;
            std::cout << "  undirected edges (edge_pairs): " << undirectedEdges << std::endl;
            std::cout << "  adjacency entries (CSR): " << totalAdjEntries << std::endl;

            // 内存占用
            auto toMB = [](size_t bytes) { return bytes / (1024.0 * 1024.0); };

            size_t endPointsBytes = totalAdjEntries * sizeof(size_t);
            size_t idMapDataBytes = id_start_point_map.size() * (sizeof(labeltype) + sizeof(size_t));
            size_t offsetMapDataBytes = offset_map.size() * (sizeof(labeltype) + sizeof(unsigned int));
            size_t edgePairsBytes = edge_pairs.capacity() * sizeof(std::pair<int,int>);

            size_t totalApproxBytes = endPointsBytes + idMapDataBytes + offsetMapDataBytes + edgePairsBytes;

            std::cout << "  memory usage (approx.):" << std::endl;
            std::cout << "    end_points: " << endPointsBytes << " B (" << toMB(endPointsBytes) << " MB)" << std::endl;
            std::cout << "    id_start_point_map data: " << idMapDataBytes << " B (" << toMB(idMapDataBytes) << " MB) [no container overhead]" << std::endl;
            std::cout << "    offset_map data: " << offsetMapDataBytes << " B (" << toMB(offsetMapDataBytes) << " MB) [no container overhead]" << std::endl;
            std::cout << "    edge_pairs capacity: " << edgePairsBytes << " B (" << toMB(edgePairsBytes) << " MB)" << std::endl;
            std::cout << "  total approx: " << totalApproxBytes << " B (" << toMB(totalApproxBytes) << " MB)" << std::endl;

            // 哈希表负载信息
            std::cout << "  id_start_point_map: size=" << id_start_point_map.size()
                      << ", buckets=" << id_start_point_map.bucket_count()
                      << ", load_factor=" << id_start_point_map.load_factor() << std::endl;
            std::cout << "  offset_map: size=" << offset_map.size()
                      << ", buckets=" << offset_map.bucket_count()
                      << ", load_factor=" << offset_map.load_factor() << std::endl;
        }

    private:
        template<typename RNG>
        size_t samplePowerLawInt(size_t min_value, size_t max_value, double tau, RNG& rng) {
            if (max_value <= min_value) return min_value;
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            double lo = static_cast<double>(min_value);
            double hi = static_cast<double>(max_value);
            double u = uniform(rng);
            if (std::fabs(tau - 1.0) < 1e-9) {
                return static_cast<size_t>(std::round(lo * std::pow(hi / lo, u)));
            }
            double a = 1.0 - tau;
            double x = std::pow(u * (std::pow(hi, a) - std::pow(lo, a)) + std::pow(lo, a), 1.0 / a);
            size_t value = static_cast<size_t>(std::round(x));
            return std::max(min_value, std::min(max_value, value));
        }

        bool hasEdge(const std::vector<std::vector<size_t>>& edges, size_t u, size_t v) const {
            const std::vector<size_t>& adj = edges[u];
            return std::find(adj.begin(), adj.end(), v) != adj.end();
        }

        void addUndirectedEdge(std::vector<std::vector<size_t>>& edges, size_t u, size_t v) {
            edges[u].push_back(v);
            edges[v].push_back(u);
        }

        void buildFromAdjacency(size_t* ids, const std::vector<std::vector<size_t>>& edges) {
            size_t num_ids = edges.size();
            size_t totalEdges = 0;
            for (size_t i = 0; i < num_ids; ++i) {
                totalEdges += edges[i].size();
                for (size_t v : edges[i]) {
                    if (i < v) edge_pairs.push_back(std::make_pair((int)ids[i], (int)ids[v]));
                }
            }

            end_points = new size_t[totalEdges];
            size_t pos = 0;
            for (size_t i = 0; i < num_ids; i++) {
                labeltype id = ids[i];
                id_start_point_map[id] = pos;
                offset_map[id] = static_cast<unsigned int>(edges[i].size());
                for (size_t v : edges[i]) {
                    end_points[pos++] = ids[v];
                }
            }
        }
    };
}
