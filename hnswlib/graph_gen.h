#pragma once
#include "hnswlib.h"
#include <unordered_set>

namespace hnswlib {

    //todo: generate a interface
    class GraphRelationSampler {
    public:
        float prob;
        std::unordered_map<labeltype, size_t> id_start_point_map;
        std::unordered_map<labeltype, unsigned int> offset_map;
        size_t * end_points{nullptr};
        // endpoints of id1 + endpoints of id2 + ...

        GraphRelationSampler(float prob):prob(prob){}

        void genRelation(size_t* ids){
            // todo:
            // generate graph information upon ids
            // distribution:
            // https://www.cnblogs.com/orion-orion/p/16254923.html 
            // Gnp
        }
    
        void saveRelation(const std::string & location) {
            // todo:
            // save sampled graph information
            // refer to:
            // hnswalg.h/HerarchicalNSW->saveIndex

        }

        void loadRelation(const std::string & location) {
            // todo:
            // load sampled graph information
            // refer to:
            // hnswalg.h/HerarchicalNSW->loadIndex
        }

    };
}