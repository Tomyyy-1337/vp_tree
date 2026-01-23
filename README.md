# Vantage-Point Tree (VP-Tree) 

A VpTree is a data structure that enables efficient searches for nearest neighbor, 
k-nearest neighbors, and radius searches in metric spaces. 

The VpTree requires stored elements to implement the Distance trait to themselves.
Additionally, search targets are required to implement Distance to the stored type.

While constructing the tree takes longer than a naive linear search,
nearest neighbors and radius searches are significantly faster using the VpTree, 
resulting in overall performance gains for multiple searches on the same dataset. 
 
## Basic Example 
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

let vp_tree = VpTree::new(points);

let target = Point { x: 2.1, y: 2.5 };

let nearest_neighbor = vp_tree.search_nearest_neighbor(&target);
assert_eq!(nearest_neighbor.unwrap(), &Point { x: 2.0, y: 2.0 });

let k_nearest = vp_tree.search_k_closest_sorted(&target, 2).collect::<Vec<_>>();
assert_eq!(k_nearest, vec![&Point { x: 2.0, y: 2.0 }, &Point { x: 3.0, y: 3.0 }]);

let radius_neighbors = vp_tree.search_in_radius_sorted(&target, 1.0).collect::<Vec<_>>();
assert_eq!(radius_neighbors, vec![&Point { x: 2.0, y: 2.0 }]);
```

## Example with different types for storage and search target
```rust
use vp_tree::*;

struct DataPoint {
   nt: Point,
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

let nearest_neighbor = vp_tree.search_nearest_neighbor(&search_point);
assert_eq!(nearest_neighbor.unwrap().data, "C".to_string());

let k_nearest = vp_tree.search_k_closest_sorted(&search_point, 2).collect::<Vec<_>>();
assert_eq!(k_nearest[0].data, "C".to_string());

let radius_neighbors = vp_tree.search_in_radius_sorted(&search_point, 1.0).collect::<Vec<_>>();
assert_eq!(radius_neighbors[0].data, "C".to_string());
```

