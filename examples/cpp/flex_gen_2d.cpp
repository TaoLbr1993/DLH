#include "../../hnswlib/hnswlib.h"
#include <random>
#include <unordered_set>

int main() {
    float p = 0.1;

    // Initing sampler
    hnswlib::Point2DSampler* psampler = new hnswlib::Point2DSampler(0.0,1.0,0.0,1.0);
    
    size_t * ids = new size_t[10];
    std::mt19937 gen(42); // Random number generator with a fixed seed
    std::uniform_int_distribution<size_t> dist(0, 100);
    std::unordered_set<size_t> unique_ids;
    while (unique_ids.size() < 10) {
        unique_ids.insert(dist(gen));
    }
    size_t i = 0;
    for (auto id : unique_ids) {
        ids[i++] = id;
        std::cout << "id: " << id << std::endl;
    }
    

    psampler->genRelation(ids, 10);
    for (const auto& pair : psampler->xcoords) {
        std::cout << "id: " << pair.first << ", x: " << pair.second
                << ", y: " << psampler->ycoords[pair.first] << std::endl;
    }
    psampler->saveRelation("grel.bin");
    psampler->loadRelation("grel.bin");
    for (const auto& pair : psampler->xcoords) {
        std::cout << "id: " << pair.first << ", x: " << pair.second
                << ", y: " << psampler->ycoords[pair.first] << std::endl;
    }
    return 0;
}
