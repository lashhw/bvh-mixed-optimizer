#ifndef BVH_MIXED_OPTIMIZER_UTILITY_HPP
#define BVH_MIXED_OPTIMIZER_UTILITY_HPP

double half_area(const float *bounds) {
    double d_x = double(bounds[1]) - double(bounds[0]);
    double d_y = double(bounds[3]) - double(bounds[2]);
    double d_z = double(bounds[5]) - double(bounds[4]);
    return (d_x + d_y) * d_z + d_x * d_y;
}

double half_area(const bvh::BoundingBox<float> &bbox) {
    float bounds[6] = { bbox.min[0], bbox.max[0], bbox.min[1], bbox.max[1], bbox.min[2], bbox.max[2] };
    return half_area(bounds);
}

void update_bbox(bvh::Bvh<float> &bvh, int idx) {
    int left_idx = bvh.nodes[idx].first_child_or_primitive;
    int right_idx = left_idx + 1;
    bvh.nodes[idx].bounding_box_proxy() = bvh::BoundingBox<float>::empty();
    bvh.nodes[idx].bounding_box_proxy().extend(bvh.nodes[left_idx].bounding_box_proxy());
    bvh.nodes[idx].bounding_box_proxy().extend(bvh.nodes[right_idx].bounding_box_proxy());
}

void serialize(const bvh::Bvh<float> &bvh,
               const std::vector<bvh::Triangle<float>> &triangles,
               const char *file_path) {
    std::ofstream out_file(file_path, std::ios::out | std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(&bvh.node_count), sizeof(size_t));
    size_t triangles_size = triangles.size();
    out_file.write(reinterpret_cast<const char*>(&triangles_size), sizeof(size_t));
    out_file.write(reinterpret_cast<const char*>(bvh.nodes.get()), sizeof(bvh::Bvh<float>::Node) * bvh.node_count);
    out_file.write(reinterpret_cast<const char*>(bvh.primitive_indices.get()), sizeof(size_t) * triangles_size);
    out_file.write(reinterpret_cast<const char*>(triangles.data()), sizeof(bvh::Triangle<float>) * triangles_size);
}

void deserialize(bvh::Bvh<float> &bvh,
                 std::vector<bvh::Triangle<float>> &triangles,
                 const char *file_path) {
    std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
    in_file.read(reinterpret_cast<char*>(&bvh.node_count), sizeof(size_t));
    size_t triangles_size;
    in_file.read(reinterpret_cast<char*>(&triangles_size), sizeof(size_t));
    triangles.resize(triangles_size);
    bvh.nodes = std::make_unique<bvh::Bvh<float>::Node[]>(bvh.node_count);
    in_file.read(reinterpret_cast<char*>(bvh.nodes.get()), sizeof(bvh::Bvh<float>::Node) * bvh.node_count);
    bvh.primitive_indices = std::make_unique<size_t[]>(triangles_size);
    in_file.read(reinterpret_cast<char*>(bvh.primitive_indices.get()), sizeof(size_t) * triangles_size);
    in_file.read(reinterpret_cast<char*>(triangles.data()), sizeof(bvh::Triangle<float>) * triangles_size);
}

#endif //BVH_MIXED_OPTIMIZER_UTILITY_HPP
