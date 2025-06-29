#pragma once
#include "hnswlib.h"
#include <unordered_set>

namespace hnswlib {

    //todo: generate a interface
    class Point2DSampler {
    public:
        float lrange, rrange, urange, drange;
        
        size_t * coords{nullptr};
        // id1 x-coor, id1 y-coor, id2 x-coor, id2 y-coor ...

        Point2DSampler(float lrange, float rrange, float drange, float urange):lrange(lrange),rrange(rrange),urange(urange),drange(drange){}

        void genRelation(size_t * ids){
            // todo:
            // generate 2D point information upon ids
            // distribution: 1-D uniform
        }
    
        void saveRelation(const std::string & location) {
            // todo:
            // save sampled 2D point information
            // refer to:
            // hnswalg.h/HerarchicalNSW->saveIndex

        }

        void loadRelation(const std::string & location) {
            // todo:
            // load sampled 2D point information
            // refer to:
            // hnswalg.h/HerarchicalNSW->loadIndex
        }
    };
}