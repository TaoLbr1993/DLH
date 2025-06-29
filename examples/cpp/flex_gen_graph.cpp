#include "../../hnswlib/hnswlib.h"

int main() {
    float p = 0.1;

    // Initing sampler
    hnswlib::GraphRelationSampler* gsampler = new hnswlib::GraphRelationSampler(p);
    
    size_t * ids = new size_t[1000];
    gsampler->genRelation(ids);
    gsampler->saveRelation("grel.bin");
    gsampler->loadRelation("grel.bin");
    return 0;
}
