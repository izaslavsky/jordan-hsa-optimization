# Spatial Unit Construction Method Comparison

## Overview

This analysis compares four spatial unit construction methods for delineating Hospital Service Areas (HSAs). **For fair comparison, baseline methods use all candidate facilities**, not just the optimized anchors. Results are reported for both the INF and NCD networks.

Use `compare_delineations.ipynb` to run `compare_spatial_methods_v2.py` with notebook-specified inputs and generate the comparison table.

---

## Methods Compared

1. **Optimized HSAs**: Greedy selection of 17 facilities with adaptive radii (10-18 km based on urban/rural setting), optimizing for population coverage, overlap minimization, and climate diversity.

2. **Fixed-radius buffers**: Circular buffers of constant radius around ALL 184 facilities.

3. **Voronoi tessellation**: Space partitioning where each point is assigned to the nearest facility (ALL 184 facilities).

4. **Governorate boundaries**: Administrative units (12 governorates).

---

## Results

| Method | Units | Coverage (%) | Overlap (%) | Multiplier | Compactness | Area (km²) | Concordance (%) |
|--------|-------|--------------|-------------|------------|-------------|------------|-----------------|
| **Optimized HSA** | **17** | **90.6** | **30.9** | **1.31x** | **0.999** | **729** | **100** |
| Fixed 10km | 184 | 90.5 | 1,419 | 15.2x | 0.988 | 309 | 88.1 |
| Fixed 15km | 184 | 95.5 | 2,353 | 24.5x | 0.982 | 692 | 81.9 |
| Fixed 18km | 184 | 96.9 | 2,994 | 30.9x | 0.977 | 991 | 78.2 |
| Fixed 20km | 184 | 97.3 | 3,449 | 35.5x | 0.973 | 1,218 | 76.0 |
| Voronoi | 154 | 99.7 | 0.0 | 1.00x | 0.655 | 582 | 10.4 |
| Governorate | 12 | 99.7 | 0.0 | 1.00x | 0.510 | 7,470 | 10.4 |

---

## Key Metrics

- **Coverage (%)**: Population within at least one spatial unit
- **Overlap (%)**: Population counted multiple times (% of unique population)
- **Multiplier**: Average times each person is counted (1.0 = no double-counting)
- **Compactness**: Shape regularity (1.0 = perfect circle)
- **Area**: Mean area per unit (km²)
- **Concordance (%)**: Area overlap with optimized HSA coverage

---

## Key Findings

### 1. Double-Counting Problem

| Method | Coverage Multiplier | Interpretation |
|--------|---------------------|----------------|
| Optimized HSA | 1.31x | Minimal overlap, manageable |
| Fixed 10km | 15.2x | Each person counted 15 times |
| Fixed 18km | 30.9x | Each person counted 31 times |
| Fixed 20km | 35.5x | Each person counted 36 times |
| Voronoi | 1.00x | No overlap (by definition) |
| Governorate | 1.00x | No overlap (by definition) |

**Critical issue**: Fixed-radius buffers using all facilities produce massive double-counting (15-36x), making disease rate calculations impossible without gravity-model allocation.

### 2. Granularity

| Method | Units | Issue |
|--------|-------|-------|
| Voronoi | 154 | Too fine for disease surveillance aggregation |
| Optimized HSA | 17 | Appropriate for regional surveillance |
| Governorate | 12 | Too coarse for local disease patterns |

### 3. Shape Quality

| Method | Compactness | Interpretation |
|--------|-------------|----------------|
| Optimized HSA | 0.999 | Nearly perfect circles (adaptive radii) |
| Fixed-radius | 0.98 | Circular (by design) |
| Voronoi | 0.655 | Irregular, elongated shapes |
| Governorate | 0.510 | Irregular administrative shapes |

### 4. Spatial Concordance

Only 10.4% of Voronoi and Governorate area overlaps with optimized HSA coverage. This indicates that:
- Voronoi tessellation does not respect healthcare delivery patterns
- Administrative boundaries do not align with healthcare facility catchments
- Optimized HSAs capture fundamentally different spatial relationships

---

## Comparison Summary

| Criterion | Optimized HSA | Fixed-radius | Voronoi | Governorate |
|-----------|---------------|--------------|---------|-------------|
| Coverage | High (91%) | High (90-97%) | Complete | Complete |
| Overlap | Low (1.31x) | Extreme (15-36x) | None | None |
| Units | 17 | 184 | 154 | 12 |
| Shape quality | Excellent | Excellent | Poor | Poor |
| Healthcare alignment | Yes | No | No | No |
| Facility selection | Optimized | None | None | None |
| Usable for surveillance | Yes | No (without allocation) | No (too fine) | No (too coarse) |

---

## Conclusion

The optimized HSA method provides:

1. **Efficient coverage**: 90.6% population coverage with only 17 units
2. **Manageable overlap**: 1.31x multiplier vs. 15-36x for fixed-radius
3. **Appropriate granularity**: 17 units vs. 154 (Voronoi) or 12 (Governorate)
4. **Compact shapes**: 0.999 compactness (near-perfect circles)
5. **Healthcare alignment**: Facilities selected based on patient volume and geographic spread

Alternative methods either produce unmanageable double-counting (fixed-radius), excessive fragmentation (Voronoi), or insufficient granularity (Governorate).

---

*Analysis: INF network, 184 candidate facilities, 17 optimized anchors*
*Script: compare_spatial_methods_v2.py*
*Output: out/{network}_spatial_methods_comparison.csv*

---

## NCD Network Results

| Method | Units | Coverage (%) | Overlap (%) | Multiplier | Compactness | Area (km²) | Concordance (%) |
|--------|-------|--------------|-------------|------------|-------------|------------|-----------------|
| **Optimized HSA** | **18** | **90.0** | **27.8** | **1.28x** | **0.999** | **725** | **100.0** |
| Fixed 10km | 195 | 91.6 | 1,557.3 | 16.57x | 0.985 | 308 | 90.2 |
| Fixed 15km | 195 | 96.0 | 2,577.4 | 26.77x | 0.979 | 688 | 84.6 |
| Fixed 18km | 195 | 97.3 | 3,199.5 | 32.99x | 0.974 | 985 | 80.3 |
| Fixed 20km | 195 | 97.7 | 3,620.1 | 37.20x | 0.970 | 1,210 | 77.7 |
| Voronoi | 195 | 99.7 | 0.0 | 1.00x | 0.650 | 460 | 11.1 |
| Governorate | 12 | 99.7 | 0.0 | 1.00x | 0.510 | 7,470 | 11.1 |

---

## NCD Key Findings

### 1. Double-Counting Problem

| Method | Coverage Multiplier | Interpretation |
|--------|---------------------|----------------|
| Optimized HSA | 1.28x | Minimal overlap, manageable |
| Fixed 10km | 16.57x | Each person counted 17 times |
| Fixed 18km | 32.99x | Each person counted 33 times |
| Fixed 20km | 37.20x | Each person counted 37 times |
| Voronoi | 1.00x | No overlap (by definition) |
| Governorate | 1.00x | No overlap (by definition) |

**Critical issue**: Fixed-radius buffers using all 195 facilities produce massive double-counting (17-37x), making disease rate calculations impossible without gravity-model allocation.

### 2. Granularity

| Method | Units | Issue |
|--------|-------|-------|
| Voronoi | 195 | Too fine for disease surveillance aggregation |
| Optimized HSA | 18 | Appropriate for regional NCD surveillance |
| Governorate | 12 | Too coarse for local NCD patterns |

### 3. Shape Quality

| Method | Compactness | Interpretation |
|--------|-------------|----------------|
| Optimized HSA | 0.999 | Nearly perfect circles (adaptive radii) |
| Fixed-radius | 0.97–0.99 | Circular (by design) |
| Voronoi | 0.650 | Irregular, elongated shapes |
| Governorate | 0.510 | Irregular administrative shapes |

### 4. Spatial Concordance

Only 11.1% of Voronoi and Governorate area overlaps with optimized HSA coverage. This indicates that:
- Voronoi tessellation does not respect healthcare delivery patterns for NCD services
- Administrative boundaries do not align with NCD facility catchments
- Optimized HSAs capture fundamentally different spatial relationships

---

## NCD Comparison Summary

| Criterion | Optimized HSA | Fixed-radius | Voronoi | Governorate |
|-----------|---------------|--------------|---------|-------------|
| Coverage | High (90%) | High (92-98%) | Complete | Complete |
| Overlap | Low (1.28x) | Extreme (17-37x) | None | None |
| Units | 18 | 195 | 195 | 12 |
| Shape quality | Excellent | Excellent | Poor | Poor |
| Healthcare alignment | Yes | No | No | No |
| Facility selection | Optimized | None | None | None |
| Usable for surveillance | Yes | No (without allocation) | No (too fine) | No (too coarse) |

---

## NCD Conclusion

The optimized HSA method for the NCD network provides:

1. **Efficient coverage**: 90.0% population coverage with only 18 units
2. **Manageable overlap**: 1.28x multiplier vs. 17-37x for fixed-radius
3. **Appropriate granularity**: 18 units vs. 195 (Voronoi) or 12 (Governorate)
4. **Compact shapes**: 0.999 compactness (near-perfect circles)
5. **Healthcare alignment**: Facilities selected based on patient volume and geographic spread

Alternative methods either produce unmanageable double-counting (fixed-radius), excessive fragmentation (Voronoi), or insufficient granularity (Governorate).

---

*Analysis: NCD network, 195 candidate facilities, 18 optimized anchors*
