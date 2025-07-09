#include "../../hnswlib/hnswlib.h"
#include <iostream>

void print_gsampler_info(const hnswlib::GraphRelationSampler* gsampler, size_t print_count = 5) {
    std::cout << "当前 GraphRelationSampler 状态：" << std::endl;
    std::cout << "  prob = " << gsampler->prob << std::endl;

    std::cout << "  id_start_point_map 大小: " << gsampler->id_start_point_map.size() << std::endl;
    std::cout << "  id_start_point_map (前" << print_count << "项): ";
    size_t cnt = 0;
    for (const auto& kv : gsampler->id_start_point_map) {
        std::cout << "{" << kv.first << ":" << kv.second << "} ";
        if (++cnt >= print_count) break;
    }
    std::cout << std::endl;

    std::cout << "  offset_map 大小: " << gsampler->offset_map.size() << std::endl;
    std::cout << "  offset_map (前" << print_count << "项): ";
    cnt = 0;
    for (const auto& kv : gsampler->offset_map) {
        std::cout << "{" << kv.first << ":" << kv.second << "} ";
        if (++cnt >= print_count) break;
    }
    std::cout << std::endl;

    // 统计 end_points 长度
    size_t totalEdges = 0;
    for (const auto& kv : gsampler->offset_map) {
        totalEdges += kv.second;
    }
    std::cout << "  end_points 总长度: " << totalEdges << std::endl;
    std::cout << "  end_points (前" << print_count << "项): ";
    for (size_t i = 0; i < std::min(print_count, totalEdges); ++i) {
        std::cout << gsampler->end_points[i] << " ";
    }
    std::cout << std::endl;
}

void test_getKHopNodes(hnswlib::GraphRelationSampler* gsampler, hnswlib::labeltype id, int k) {
    std::cout << "测试 getKHopNodes, id = " << id << ", k = " << k << std::endl;
    std::unordered_set<hnswlib::labeltype> kHopNodes = gsampler->getKHopNodes(id, k);
    std::cout << "  k-hop 节点数量: " << kHopNodes.size() << std::endl;
    std::cout << "  节点列表(前20个): ";
    int cnt = 0;
    for (auto v : kHopNodes) {
        std::cout << v << " ";
        if (++cnt >= 20) break;
    }
    std::cout << std::endl;
}

int main() {
    float p = 0.1;
    std::cout << "初始化 GraphRelationSampler，p = " << p << std::endl;

    hnswlib::GraphRelationSampler* gsampler = new hnswlib::GraphRelationSampler(p);

    size_t * ids = new size_t[1000];
    for (size_t i = 0; i < 1000; ++i) ids[i] = i;

    std::cout << "开始生成关系..." << std::endl;
    gsampler->genRelation(ids, 1000);
    std::cout << "关系生成完成" << std::endl;
    print_gsampler_info(gsampler);

    // 测试 getKHopNodes
    test_getKHopNodes(gsampler, 0, 1);
    test_getKHopNodes(gsampler, 0, 2);
    test_getKHopNodes(gsampler, 1, 3);

    std::cout << "保存关系到 grel.bin..." << std::endl;
    gsampler->saveRelation("grel.bin");
    std::cout << "保存完成" << std::endl;

    std::cout << "从 grel.bin 加载关系..." << std::endl;
    gsampler->loadRelation("grel.bin");
    std::cout << "加载完成" << std::endl;
    print_gsampler_info(gsampler);

    // 再次测试 getKHopNodes
    test_getKHopNodes(gsampler, 0, 1);
    test_getKHopNodes(gsampler, 0, 2);
    test_getKHopNodes(gsampler, 1, 3);

    delete[] ids;
    delete gsampler;
    std::cout << "程序结束" << std::endl;
    return 0;
}