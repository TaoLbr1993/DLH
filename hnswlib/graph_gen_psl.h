#pragma once
#include "hnswlib.h"
#include <unordered_set>
#include <random>
#include <vector>
#include <unordered_set>

namespace hnswlib {

    //todo: generate a interface
    class GraphRelationSamplerPLL {
    public:
        float prob;
        std::vector<std::pair<labeltype, labeltype>> edges;
        std::unordered_map<labeltype, size_t> id_start_point_map;
        std::unordered_map<labeltype, unsigned int> offset_map;
        size_t * end_points{nullptr};
        // endpoints of id1 + endpoints of id2 + ...

        GraphRelationSamplerPLL(float prob):prob(prob){}

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
                        edges.push_back(std::make_pair(ids[i], ids[j]));
                    }
                }
            }
        }
    };
}