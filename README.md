# Vantage-Point Tree (VP-Tree) 

A `VpTree` is a data structure that enables efficient nearest neighbor, 
k-nearest neighbors, and radius searches in metric spaces with arbitrary dimensionality.

Why use this [crate](https://crates.io/crates/vp_tree)? The implementation is focused on speed and easy to use with all types by implementing a simple trait. In my benchmarks, it outperforms the "vpsearch" and "vec-vp-tree" crates by a significant margin for building with similar performance for searching.

The implementation is generic over any type that implements the `Distance` trait.
The `VpTree` requires stored elements to implement the `Distance` in relation to themselves. 
For faster tree construction, it is recommended to also implement the `Distance::distance_heuristic` method (for example squared distance to avoid square root calculations) to accelerate tree construction.

For large datasets, the tree can be constructed using multiple threads.

For searching in the `VpTree`, either the stored type or a different type that implements the `Distance` trait in relation to the stored type can be used.

While constructing the tree takes longer than a naive linear search,
nearest neighbor and radius searches are significantly faster using the `VpTree`, 
resulting in overall performance gains for multiple searches on the same dataset. 
Larger datasets benefit more from the VP-Tree structure.

## Example 
```rust
use vp_tree::*;

#[derive(Debug, PartialEq)]
struct Point {
   x: f64,
   y: f64,
}

impl Distance<Point> for Point {
   fn distance_heuristic(&self, other: &Point) -> f64 {
      let dx = self.x - other.x;
      let dy = self.y - other.y;
      dx * dx + dy * dy
   }

   fn distance(&self, other: &Point) -> f64 {
      self.distance_heuristic(other).sqrt()
   }
}

let points = vec![
   Point { x: 0.0, y: 0.0 },
   Point { x: 1.0, y: 1.0 },
   Point { x: 2.0, y: 2.0 },
   Point { x: 3.0, y: 3.0 },
];

// Build VpTree storing references to the original points 
let _vp_tree_index = VpTree::new_index(&points);

// Build VpTree storing owned data on 8 threads 
// If possible, prefere building the tree with owned data for best performance
let vp_tree = VpTree::new_parallel(points, 8); 

let target = Point { x: 2.1, y: 2.5 };

let nearest_neighbor = vp_tree.nearest_neighbor(&target);
assert_eq!(nearest_neighbor.unwrap(), &Point { x: 2.0, y: 2.0 });

let k_nearest = vp_tree.querry(&target, Querry::k_nearest_neighbors(2).sorted());
assert_eq!(k_nearest, vec![&Point { x: 2.0, y: 2.0 }, &Point { x: 3.0, y: 3.0 }]);

let radius_neighbors = vp_tree.querry(&target, Querry::neighbors_within_radius(1.0).sorted());
assert_eq!(radius_neighbors, vec![&Point { x: 2.0, y: 2.0 }]);
```

## Example with different types for storage and search target
```rust
use vp_tree::*;

struct DataPoint {
   point: Point,
   data: String,
}
impl Distance<DataPoint> for DataPoint {
    fn distance_heuristic(&self, other: &DataPoint) -> f64 {
        let dx = self.point.x - other.point.x;
        let dy = self.point.y - other.point.y;
        dx * dx + dy * dy
    }
    fn distance(&self, other: &DataPoint) -> f64 {
        self.distance_heuristic(other).sqrt()
    }
}
struct Point {
    x: f64,
    y: f64,
}
impl Distance<DataPoint> for Point {
    fn distance(&self, other: &DataPoint) -> f64 {
        let dx = self.x - other.point.x;
        let dy = self.y - other.point.y;
        ((dx * dx) + (dy * dy)).sqrt()
   }
}    

let data_points = vec![
    DataPoint { point: Point { x: 0.0, y: 0.0 }, data: "A".to_string() },
    DataPoint { point: Point { x: 1.0, y: 1.0 }, data: "B".to_string() },
    DataPoint { point: Point { x: 2.0, y: 2.0 }, data: "C".to_string() },
    DataPoint { point: Point { x: 3.0, y: 3.0 }, data: "D".to_string() },  
];

let vp_tree = VpTree::new(data_points);
let search_point = Point { x: 2.1, y: 2.5 };

let nearest_neighbor = vp_tree.nearest_neighbor(&search_point);
assert_eq!(nearest_neighbor.unwrap().data, "C".to_string());

let k_nearest = vp_tree.querry(&search_point, Querry::k_nearest_neighbors(2).sorted());
assert_eq!(k_nearest[0].data, "C".to_string());

let radius_neighbors = vp_tree.querry(&search_point, Querry::neighbors_within_radius(1.0).sorted());
assert_eq!(radius_neighbors[0].data, "C".to_string());
```
