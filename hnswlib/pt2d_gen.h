#pragma once
#include "hnswlib.h"
#include <unordered_set>
#include <unordered_map>
#include <random>

namespace hnswlib {

    //todo: generate a interface
    class Point2DSampler {
    public:
        float lrange, rrange, urange, drange;
        std::unordered_map<labeltype, float> xcoords;
        std::unordered_map<labeltype, float> ycoords;
        // id1 x-coor, id1 y-coor, id2 x-coor, id2 y-coor ...

        Point2DSampler(float lrange, float rrange, float drange, float urange):lrange(lrange),rrange(rrange),urange(urange),drange(drange){}

        void clear() {
            xcoords.clear();
            ycoords.clear();
        }

        void genRelation(size_t * ids, size_t num_ids) {
            // todo:
            // generate 2D point information upon ids
            // distribution: 1-D uniform
            clear();
            std::default_random_engine engine(42); // fixed seed for reproducibility
            std::uniform_real_distribution<float> x_dist(lrange, rrange);
            std::uniform_real_distribution<float> y_dist(drange, urange);
            for (size_t i = 0; i < num_ids; ++i) {
                labeltype id = static_cast<labeltype>(ids[i]);
                float x = x_dist(engine);
                float y = y_dist(engine);
                xcoords[id] = x;
                ycoords[id] = y;
                std::cout << "gen id: " << id << ", x: " << x << ", y: " << y << std::endl;
            }
        }
    
        void saveRelation(const std::string & location) {
            // todo:
            // save sampled 2D point information
            // refer to:
            // hnswalg.h/HerarchicalNSW->saveIndex
            std::ofstream output(location, std::ios::binary);
            if (!output.is_open()) {
                throw std::runtime_error("Cannot open file for saving relation");
            }
            writeBinaryPOD(output, lrange);
            writeBinaryPOD(output, rrange);
            writeBinaryPOD(output, drange);
            writeBinaryPOD(output, urange);
            size_t num_points = xcoords.size();
            writeBinaryPOD(output, num_points);
            for (const auto & pair : xcoords) {
                labeltype id = pair.first;
                float x = pair.second;
                float y = ycoords[id];
                writeBinaryPOD(output, id);
                writeBinaryPOD(output, x);
                writeBinaryPOD(output, y);
            }
            output.close();
        }

        void loadRelation(const std::string & location) {
            // todo:
            // load sampled 2D point information
            // refer to:
            // hnswalg.h/HerarchicalNSW->loadIndex

            std::ifstream input(location, std::ios::binary);
            if (!input.is_open()) {
                throw std::runtime_error("Cannot open file for loading relation");
            }
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);
            std::streampos position;
            readBinaryPOD(input, lrange);
            readBinaryPOD(input, rrange);
            readBinaryPOD(input, drange);
            readBinaryPOD(input, urange);
            size_t num_points;
            readBinaryPOD(input, num_points);
            if (num_points * sizeof(labeltype) + sizeof(size_t) + num_points * 2 * sizeof(float) + 4 * sizeof(float) != total_filesize) {
                throw std::runtime_error("File size does not match expected size for loaded relation");
            }
            for (size_t i = 0; i < num_points; ++i) {
                labeltype id;
                float x, y;
                readBinaryPOD(input, id);
                readBinaryPOD(input, x);
                readBinaryPOD(input, y);
                xcoords[id] = x;
                ycoords[id] = y;
        }
            input.close();
        }
    };
}