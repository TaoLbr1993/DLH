#include "../../hnswlib/hnswlib.h"
#include <iostream>

int main() {
    std::priority_queue<hnswlib::HopNbrTrible<float>> q;
    q.emplace(true, 1.0, 0);
    q.emplace(true, 0.8, 1);
    q.emplace(false, 1.5, 2);
    q.emplace(false, 0.2, 3);

    while (!q.empty()) {
        hnswlib::HopNbrTrible<float> t = q.top();
        q.pop();
        std::cout << t.dist << " " << t.id << " " << t.in_hop << std::endl;
    }
}
