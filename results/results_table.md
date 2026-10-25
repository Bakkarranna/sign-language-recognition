# Sign language recognition - results

| model | top-1 accuracy | macro F1 | top-5 | n |
| --- | --- | --- | --- | --- |
| SignGRU (biGRU, 2x256, 32-frame landmarks) | 0.1111 | 0.0822 | 0.2963 | 162 |
| nearest-mean (cosine, temporal mean) | 0.0370 | 0.0188 | 0.1914 | 162 |
| per-frame RandomForest (200 trees, mean vote) | 0.1728 | 0.1327 | 0.3580 | 162 |
