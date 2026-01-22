use vp_tree::*;

#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Distance<Point> for Point {
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn main() {
    let num_points = 1_000_000;

    let random_points = (0..num_points)
        .map(|_| Point {
            x: fastrand::f64() * 1000.0,
            y: fastrand::f64() * 1000.0,
        })
        .collect::<Vec<_>>();

    let target_point = Point { x: 500.0, y: 500.0 };

    println!("Baseline linear search:");

    let start = std::time::Instant::now();
    let nearest_linear = find_nearest_neighbor_linear(&random_points, &target_point);
    let baseline_duration = start.elapsed();
    println!("Time taken for linear search with {} points: {:?}, Result: {:?}", num_points, baseline_duration, nearest_linear);

    let start = std::time::Instant::now();
    let k_closest_linear = find_k_closest_linear(&random_points, &target_point, 5);
    let k_baseline_duration = start.elapsed();
    println!("Time taken to find 5 closest neighbors linearly: {:?}. Result count: {}", k_baseline_duration, k_closest_linear.len());

    let start = std::time::Instant::now();
    let in_radius_linear = find_in_radius_linear(&random_points, &target_point, 5.0);
    let radius_baseline_duration = start.elapsed();
    println!("Time taken to find points within radius 5.0 linearly: {:?}, found {} points", radius_baseline_duration, in_radius_linear.len());
    
    println!("\nVpTree search:");

    let start = std::time::Instant::now();
    let vp_tree = vp_tree::VpTree::new(random_points);
    let duration = start.elapsed();
    println!("Time taken to build VpTree with {} points: {:?}", num_points, duration);

    let start = std::time::Instant::now();
    let nearest_neighbor = vp_tree.search_nearest_neighbor(&target_point);
    let duration = start.elapsed();
    println!("Time taken to search nearest neighbor: {:?}, {:.2?} times faster than linear search. Result: {:?}", duration, baseline_duration.as_secs_f64() / duration.as_secs_f64(), nearest_neighbor);

    let start = std::time::Instant::now();
    let k_closest_neighbors = vp_tree.search_k_closest(&target_point, 5);
    let duration = start.elapsed();
    println!("Time taken to search 5 closest neighbors: {:?}, {:.2?} times faster than linear search. Result count: {}", duration, k_baseline_duration.as_secs_f64() / duration.as_secs_f64(), k_closest_neighbors.count());

    let start = std::time::Instant::now();
    let in_radius = vp_tree.search_in_radius(&target_point, 5.0);
    let duration = start.elapsed();
    println!("Time taken to search points within radius 5.0: {:?}, {:.2?} times faster than linear search. Result count: {}", duration, radius_baseline_duration.as_secs_f64() / duration.as_secs_f64(), in_radius.count());
}

fn find_nearest_neighbor_linear<'a>(points: &'a Vec<Point>, target: &Point) -> Option<&'a Point> {
    points.iter().min_by(|a, b| {
        let dist_a = a.distance(target);
        let dist_b = b.distance(target);
        dist_a.partial_cmp(&dist_b).unwrap()
    })
}

fn find_k_closest_linear<'a>(points: &'a Vec<Point>, target: &Point, k: usize) -> Vec<&'a Point> {
    let mut points_with_distance: Vec<(&Point, f64)> = points
        .iter()
        .map(|p| (p, p.distance(target)))
        .collect();
    
    points_with_distance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    points_with_distance.iter().take(k).map(|(p, _)| *p).collect()
}

fn find_in_radius_linear<'a>(points: &'a Vec<Point>, target: &Point, radius: f64) -> Vec<&'a Point> {
    points
        .iter()
        .filter(|p| p.distance(target) <= radius)
        .collect()
}