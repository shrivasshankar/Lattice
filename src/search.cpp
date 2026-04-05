#include "search.h"

#include <queue>
#include <algorithm>
#include <stdexcept>

namespace lattice {

std::vector<SearchResult> brute_force_knn(
    const VectorDataset& dataset,
    const float* query,
    uint32_t k,
    DistanceFn distance_fn
) {
    if (k == 0) return {};
    if (dataset.num_vectors == 0) return {};

    // Max-heap: the top element is always the WORST (farthest) of our
    // current k best candidates. This lets us efficiently decide whether
    // a new vector is worth keeping — if it's closer than the worst,
    // we pop the worst and push the new one.
    auto cmp = [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;  // max-heap: largest distance on top
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> max_heap(cmp);

    for (uint32_t i = 0; i < dataset.num_vectors; ++i) {
        float dist = distance_fn(dataset.get_vector(i), query, dataset.dimension);

        if (max_heap.size() < k) {
            max_heap.push({i, dist});
        } else if (dist < max_heap.top().distance) {
            max_heap.pop();
            max_heap.push({i, dist});
        }
    }

    // Extract results and sort closest-first
    std::vector<SearchResult> results;
    results.reserve(max_heap.size());
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });

    return results;
}

} // namespace lattice
