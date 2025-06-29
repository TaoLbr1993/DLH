#include "../../hnswlib/hnswlib.h"

int main() {
    float p = 0.1;

    // Initing sampler
    hnswlib::Point2DSampler* psampler = new hnswlib::Point2DSampler(0.0,1.0,0.0,1.0);
    
    size_t * ids = new size_t[1000];
    psampler->genRelation(ids);
    psampler->saveRelation("grel.bin");
    psampler->loadRelation("grel.bin");
    return 0;
}
