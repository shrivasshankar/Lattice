float l2_distance(const float*a, const float*b, uint32_t dimension){
    float sum = 0.0f;
    for (uint32_t i = 0; i < dimension; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float cosine_distance(const float* a, const float* b, uint32_t dimension){
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t i = 0; i < dimension; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom == 0.0f) return 0.0f;

    return 1.0f - (dot / denom);
}