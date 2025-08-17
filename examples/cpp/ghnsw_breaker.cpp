# include "../../hnswlib/hnswlib.h"

int main() {
    std::vector<int> test_cases = {1,2,3,4,5,6,7};
    // one breaker
    std::cout << "one breaker test" << std::endl;
    for (auto tc: test_cases) {
        std::vector<int> res = hnswlib::one_range_breaker(tc);
        std:: cout << "case: " << tc << std::endl;
        for (auto ri: res) {
            std::cout << ri << "+";
        }
        std::cout << std::endl;
    }

    // loginc breaker
    std::cout << "loginc breaker test" << std::endl;
    for (auto tc: test_cases) {
        std::vector<int> res = hnswlib::loginc_range_breaker(tc);
        std:: cout << "case: " << tc << std::endl;
        for (auto ri: res) {
            std::cout << ri << "+";
        }
        std::cout << std::endl;
    }

    // logdec breaker
    std::cout << "logdec breaker test" << std::endl;
    for (auto tc: test_cases) {
        std::vector<int> res = hnswlib::logdec_range_breaker(tc);
        std:: cout << "case: " << tc << std::endl;
        for (auto ri: res) {
            std::cout << ri << "+";
        }
        std::cout << std::endl;
    }
}