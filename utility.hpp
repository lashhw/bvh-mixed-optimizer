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

#endif //BVH_MIXED_OPTIMIZER_UTILITY_HPP
