use vp_tree::Distance;

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

fn main() {
    let random_points = (0..10_000)
        .map(|_| Point {
            x: fastrand::f64() * 1000.0,
            y: fastrand::f64() * 1000.0,
            z: fastrand::f64() * 1000.0,
        })
        .collect::<Vec<_>>();

    let target_point = Point { x: 500.0, y: 500.0, z: 500.0 };
    
    let vp_tree = vp_tree::VpTree::new(random_points);

    let _nearest_neighbor = vp_tree.search_nearest_neighbor(&target_point);
    let _k_closest_neighbors = vp_tree.search_k_closest(&target_point, 5);
    let _in_radius = vp_tree.search_in_radius(&target_point, 100.0);
}
