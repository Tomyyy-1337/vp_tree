use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use vp_tree::Distance;

const DIMENSIONS: usize = 20;

#[derive(Clone)]
struct Point<const D: usize> {
    cords: [f64; DIMENSIONS],
}

impl<const D: usize> Distance<Point<D>> for Point<D> {
    fn distance(&self, other: &Point<D>) -> f64 {
        self.distance_heuristic(other).sqrt()
    }

    fn distance_heuristic(&self, other: &Point<D>) -> f64 {
        self.cords.iter().zip(other.cords.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum()
    }
}

impl<const D: usize> Point<D> {
    fn new_random() -> Self {
        Point {
            cords: [(); DIMENSIONS].map(|_| fastrand::f64() * 1000.0),
        }
    }
}

fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Construction");

    let num_points = [10_000, 1_000_000];
    let num_threads = [1, 4, 16];

    for &points in &num_points {
        for &threads in &num_threads {
            group.bench_function(format!("Constructing VpTree with {} points on {:02} threads", points, threads),
                |b|b.iter_batched(
                    || {
                        (0..points)
                        .map(|_| Point::<DIMENSIONS>::new_random())
                        .collect()
                    },
                    |data| {
                        let _vp_tree = vp_tree::VpTree::new_parallel(black_box(data), black_box(threads));
                    },
                    criterion::BatchSize::LargeInput,
                ),
            );
        }
    }
}

fn construction_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Construction (Indirect access)");

    let num_points = [10_000, 1_000_000];
    let num_threads = [1, 4, 16];

    for &points in &num_points {
        for &threads in &num_threads {
            group.bench_function(format!("Constructing VpTree with {} points on {:02} threads (Indirect access)", points, threads),
                |b|b.iter_batched(
                    || {
                        (0..points)
                        .map(|_| Point::<DIMENSIONS>::new_random())
                        .collect::<Vec<Point<DIMENSIONS>>>()
                    },
                    |data| {
                        let _vp_tree = vp_tree::VpTree::new_index_parallel(black_box(&data), black_box(threads));
                    },
                    criterion::BatchSize::LargeInput,
                ),
            );
        }
    }
}

fn nearest_neighbor_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Nearest Neighbor Search");

    let num_points = [10_000, 100_000, 1_000_000, 10_000_000];

    for &points in &num_points {
        let random_points: Vec<Point<DIMENSIONS>    > = (0..points)
            .map(|_| Point::new_random())
            .collect();

        let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

        group.bench_function(format!("Nearest neighbor search in VpTree with {} points", points),
            |b| b.iter_batched(
                || Point::new_random(),
                |target| {
                    let _nn = vp_tree.nearest_neighbor(black_box(&target));
                },
                criterion::BatchSize::SmallInput,
            ),
        );
    }
}

fn nearest_neighbor_search_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Nearest Neighbor Search (Indirect access)");

    let num_points = [10_000, 100_000, 1_000_000, 10_000_000];

    for &points in &num_points {
        let random_points: Vec<Point<DIMENSIONS>> = (0..points)
            .map(|_| Point::<DIMENSIONS>::new_random())
            .collect::<Vec<Point<DIMENSIONS>>>();

        let vp_tree = vp_tree::VpTree::new_index_parallel(&random_points, 16);

        group.bench_function(format!("Nearest neighbor search in VpTree with {} points", points),
            |b| b.iter_batched(
                || Point::new_random(),
                |target| {
                    let _nn = vp_tree.nearest_neighbor(black_box(&target));
                },
                criterion::BatchSize::SmallInput,
            ),
        );
    }
}

fn k_nearest_neighbors_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree K Nearest Neighbors Search");

    let num_points = [10_000, 100_000, 1_000_000];
    let ks = [1, 5, 10, 50];

    for &points in &num_points {
        for k in &ks {
            let random_points: Vec<Point<DIMENSIONS>> = (0..points)
                .map(|_| Point::new_random())
                .collect();

            let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

            group.bench_function(format!("K={} nearest neighbors search in VpTree with {} points", k, points),
                |b| b.iter_batched(
                    || Point::new_random(),
                    |target| {
                        let _k_nn = vp_tree.querry(black_box(&target), black_box(vp_tree::Querry::k_nearest_neighbors(*k)));
                    },
                    criterion::BatchSize::SmallInput,
                ),
            );
        }
    }
}

fn k_nearest_neighbors_search_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree K Nearest Neighbors Search (Indirect access)");

    let num_points = [10_000, 100_000, 1_000_000];
    let ks = [1, 5, 10, 50];

    for &points in &num_points {
        for k in &ks {
            let random_points: Vec<Point<DIMENSIONS>> = (0..points)
                .map(|_| Point::<DIMENSIONS>::new_random())
                .collect::<Vec<Point<DIMENSIONS>>>();

            let vp_tree = vp_tree::VpTree::new_index_parallel(&random_points, 16);

            group.bench_function(format!("K={} nearest neighbors search in VpTree with {} points", k, points),
                |b| b.iter_batched(
                    || Point::new_random(),
                    |target| {
                        let _k_nn = vp_tree.querry(black_box(&target), black_box(vp_tree::Querry::k_nearest_neighbors(*k)));
                    },
                    criterion::BatchSize::SmallInput,
                ),
            );
        }
    }
}

fn radius_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Radius Search");

    let num_points = [10_000, 100_000, 1_000_000];
    let radii = [1.0, 10.0, 100.0];

    for &points in &num_points {
        for &radius in &radii {
            let random_points: Vec<Point<DIMENSIONS>> = (0..points)
                .map(|_| Point::new_random())
                .collect();

            let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

            group.bench_function(format!("Radius={} search in VpTree with {} points", radius, points),
                |b| b.iter_batched(
                    || Point::new_random(),
                    |target| {
                        let _in_radius = vp_tree.querry(black_box(&target), black_box(vp_tree::Querry::neighbors_within_radius(radius)));
                    },
                    criterion::BatchSize::SmallInput,
                ),
            );
        }
    }
}

fn radius_search_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("VpTree Radius Search (Indirect access)");

    let num_points = [10_000, 100_000, 1_000_000];
    let radii = [1.0, 10.0, 100.0];

    for &points in &num_points {
        for &radius in &radii {
            let random_points: Vec<Point<DIMENSIONS>> = (0..points)
                .map(|_| Point::new_random())
                .collect();

            let vp_tree = vp_tree::VpTree::new_index_parallel(&random_points, 16);

            group.bench_function(format!("Radius={} search in VpTree with {} points", radius, points),
                |b| b.iter_batched(
                    || Point::new_random(),
                    |target| {
                        let _in_radius = vp_tree.querry(black_box(&target), black_box(vp_tree::Querry::neighbors_within_radius(radius)));
                    },
                    criterion::BatchSize::SmallInput,
                ),
            );
        }
    }
}

criterion_group!(benches1, construction, construction_index);
criterion_group!(benches2, nearest_neighbor_search, nearest_neighbor_search_index);
criterion_group!(benches3, k_nearest_neighbors_search, k_nearest_neighbors_search_index);
criterion_group!(benches4, radius_search, radius_search_index);

criterion_main!(benches1, benches2, benches3, benches4);