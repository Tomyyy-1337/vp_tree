use vp_tree::Distance;

struct DataPoint {
    x: f64,
    y: f64,
    _data: String,
}

struct Point {
    x: f64,
    y: f64,
}

impl Distance<DataPoint> for DataPoint {
    fn distance(&self, other: &DataPoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Distance<DataPoint> for Point {
    fn distance(&self, other: &DataPoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn main() {
    let random_points = (0..10_000)
        .map(|i| DataPoint {
            x: fastrand::f64() * 1000.0,
            y: fastrand::f64() * 1000.0,
            _data: format!("Point {}", i),
        })
        .collect::<Vec<_>>();

    let vp_tree = vp_tree::VpTree::new(random_points);

    let target_point = Point { x: 500.0, y: 500.0 };

    let _nearest_neighbor = vp_tree.search_nearest_neighbor(&target_point);
    let _k_closest_neighbors = vp_tree.search_closest_k(&target_point, 5);
    let _in_radius = vp_tree.search_in_radius(&target_point, 100.0);
}