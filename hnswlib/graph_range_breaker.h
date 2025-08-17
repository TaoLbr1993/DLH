#pragma once

#include <math.h>

namespace hnswlib {
    std::vector<int> one_range_breaker(int maxr) {
        assert(maxr>0);
        std::vector<int> ret;
        for (int i=0; i<maxr; i++) ret.push_back((int)1);
        return ret;
    }

    std::vector<int> loginc_range_breaker(int maxr) {
        assert(maxr>0);
        std::vector<int> ret;
        int step = 1;
        int rangesum = 0;
        while (true) {
            if (rangesum + step <= maxr) {
                rangesum += step;
                ret.push_back(step);
                step *= 2;
            }
            else {
                ret.push_back(maxr-rangesum);
                rangesum = maxr;

            }
            if (rangesum == maxr) break;

        }
        return ret;
    }

    std::vector<int> logdec_range_breaker(int maxr) {
        assert(maxr>0);
        std::vector<int> ret;
        std::vector<int> fret;
        int step = 1;
        int rangesum = 0;
        while (true) {
            if (rangesum + step <= maxr) {
                rangesum += step;
                fret.push_back(step);
                step *= 2;
            }
            else {
                break;
            }
            if (rangesum == maxr) break;
        }
        
        for (int i=fret.size()-1; i>=0; i--) {
            ret.push_back(fret[i]);
        }
        if (rangesum<maxr) ret.push_back(maxr-rangesum);

        return ret;
    }
}