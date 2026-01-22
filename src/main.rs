use std::collections::BinaryHeap;

use vp_tree::{Distance, VpTree};

#[derive(Debug, Clone, PartialEq)]
struct DataPoint {
    point: Point,
    data: String,
}

impl Distance<DataPoint> for DataPoint {
    fn distance(&self, other: &DataPoint) -> f64 {
        ((self.point.x - other.point.x).powi(2) + (self.point.y - other.point.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

impl Distance<DataPoint> for Point {
    fn distance(&self, other: &DataPoint) -> f64 {
        ((self.x - other.point.x).powi(2) + (self.y - other.point.y).powi(2)).sqrt()
    }
}

fn test_vp_tree_construction() {
    let points = vec![
        DataPoint { point: Point { x: 0.0, y: 0.0 }, data: "A".to_string() },
        DataPoint { point: Point { x: 1.0, y: 1.0 }, data: "B".to_string() },
        DataPoint { point: Point { x: 2.0, y: 2.0 }, data: "C".to_string() },
        DataPoint { point: Point { x: 3.0, y: 3.0 }, data: "D".to_string() },
        DataPoint { point: Point { x: 4.0, y: 4.0 }, data: "E".to_string() },
        DataPoint { point: Point { x: 5.0, y: 5.0 }, data: "F".to_string() },
        DataPoint { point: Point { x: 6.0, y: 6.0 }, data: "G".to_string() },
        DataPoint { point: Point { x: 50.0, y: 50.0 }, data: "H".to_string() },
        DataPoint { point: Point { x: 51.0, y: 51.0 }, data: "I".to_string() },
        DataPoint { point: Point { x: 52.0, y: 52.0 }, data: "J".to_string() },
    ];

    let vp_tree = VpTree::new(points);

    // Search using a Point as the target
    let _result = vp_tree.search_closest_k(&Point { x: 2.5, y: 2.5 }, 3);
    // Search using a DataPoint as the target
    let seatch_data_point = DataPoint { point: Point { x: 230.0, y: 30.0 }, data: "Target".to_string() };
    let result = vp_tree.search_closest_k(&seatch_data_point, 3);

    println!("Search results:");
    for point in result {
        println!("{:?}", point);
    }   
}


fn main() {
    test_vp_tree_construction();

    let random_points = (0..10_000)
        .map(|i| DataPoint {
            point: Point {
                x: fastrand::f64() * 1000.0,
                y: fastrand::f64() * 1000.0,
            },
            data: format!("Point {}", i),
        })
        .collect::<Vec<_>>();
    let random_points_clone = random_points.clone();

    let search_point = Point { x: 500.0, y: 500.0 };
    
    let build_start = std::time::Instant::now();
    let vp_tree = VpTree::new(random_points);
    let build_duration = build_start.elapsed();
    println!("VP-Tree build time: {:?}", build_duration);

    let search_start = std::time::Instant::now();
    let result = vp_tree.search_closest_k_sorted(&search_point, 10).collect::<Vec<_>>();
    let search_duration = search_start.elapsed();
    println!("VP-Tree search time: {:?}", search_duration);

    let search_start = std::time::Instant::now();
    let baseline_result = baseline_linear_search(&random_points_clone, &search_point, 10);
    let baseline_search_duration = search_start.elapsed();
    println!("Baseline linear search time: {:?}", baseline_search_duration);
    assert_eq!(baseline_result, result);

    let nn_start = std::time::Instant::now();
    let _nn_result = vp_tree.search_nearest_neighbor(&search_point);
    let nn_duration = nn_start.elapsed();
    println!("VP-Tree nearest neighbor search time: {:?}", nn_duration);
    
    let search_in_radius_start = std::time::Instant::now();
    let radius_result = vp_tree.search_in_radius_sorted(&search_point, 5.0).collect::<Vec<_>>();
    let search_in_radius_duration = search_in_radius_start.elapsed();
    println!("VP-Tree search in radius time: {:?} results found: {}", search_in_radius_duration, radius_result.len());

}

fn baseline_linear_search<'a, T, U>(data: &'a [T], target: &U, k: usize) -> Vec<&'a T>
where
    U: Distance<T>,
{
    let mut heap = BinaryHeap::with_capacity(k);
    let mut tau = f64::INFINITY;

    for item in data {
        let dist = target.distance(item);
        if dist < tau {
            heap.push(HeapItem { distance: dist, item });
            if heap.len() > k {
                heap.pop();
            }
            if heap.len() == k {
                tau = heap.peek().unwrap().distance;
            }
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|heap_item| heap_item.item)
        .collect()
}


pub struct HeapItem<T> {
    distance: f64,
    item: T,
}

impl<T> PartialEq for HeapItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl<T> Eq for HeapItem<T> {}

impl<T> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for HeapItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}