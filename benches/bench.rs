use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use vp_tree::Distance;

#[derive(Clone)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Distance<Point> for Point {
    fn distance(&self, other: &Point) -> f64 {
        self.distance_heuristic(other).sqrt()
    }

    fn distance_heuristic(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
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
                        .map(|_| Point {
                            x: fastrand::f64() * 1000.0,
                            y: fastrand::f64() * 1000.0,
                            z: fastrand::f64() * 1000.0,
                        })
                        .collect()
                    },
                    |data| {
                        let _vp_tree = vp_tree::VpTree::new_parallel(data, threads);
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
        let random_points: Vec<Point> = (0..points)
            .map(|_| Point {
                x: fastrand::f64() * 1000.0,
                y: fastrand::f64() * 1000.0,
                z: fastrand::f64() * 1000.0,
            })
            .collect();

        let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

        group.bench_function(format!("Nearest neighbor search in VpTree with {} points", points),
            |b| b.iter_batched(
                || Point { x: fastrand::f64() * 1000.0, y: fastrand::f64() * 1000.0, z: fastrand::f64() * 1000.0 },
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
    let ks = [1, 5, 10, 50, 500, 1000];

    for &points in &num_points {
        for k in &ks {
            let random_points: Vec<Point> = (0..points)
                .map(|_| Point {
                    x: fastrand::f64() * 1000.0,
                    y: fastrand::f64() * 1000.0,
                    z: fastrand::f64() * 1000.0,
                })
                .collect();

            let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

            group.bench_function(format!("K={} nearest neighbors search in VpTree with {} points", k, points),
                |b| b.iter_batched(
                    || Point { x: fastrand::f64() * 1000.0, y: fastrand::f64() * 1000.0, z: fastrand::f64() * 1000.0 },
                    |target| {
                        let _k_nn = vp_tree.querry(black_box(&target), vp_tree::Querry::k_nearest_neighbors(*k));
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
            let random_points: Vec<Point> = (0..points)
                .map(|_| Point {
                    x: fastrand::f64() * 1000.0,
                    y: fastrand::f64() * 1000.0,
                    z: fastrand::f64() * 1000.0,
                })
                .collect();

            let vp_tree = vp_tree::VpTree::new_parallel(random_points.clone(), 16);

            group.bench_function(format!("Radius={} search in VpTree with {} points", radius, points),
                |b| b.iter_batched(
                    || Point { x: fastrand::f64() * 1000.0, y: fastrand::f64() * 1000.0, z: fastrand::f64() * 1000.0 },
                    |target| {
                        let _in_radius = vp_tree.querry(black_box(&target), vp_tree::Querry::neighbors_within_radius(radius));
                    },
                    criterion::BatchSize::SmallInput,
                ),
            );
        }
    }
}

criterion_group!(benches1, construction);
criterion_group!(benches2, nearest_neighbor_search);
criterion_group!(benches3, k_nearest_neighbors_search);
criterion_group!(benches4, radius_search);

criterion_main!(benches1, benches2, benches3, benches4);