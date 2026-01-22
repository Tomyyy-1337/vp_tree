
#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use crate::*;

    #[test]
    fn test_nn() {
        #[derive(Debug)]
        struct TestPoint {
            value: f64,
        }

        impl Distance<TestPoint> for TestPoint {
            fn distance(&self, other: &TestPoint) -> f64 {
                (self.value - other.value).abs()
            }
        }

        let points = vec![
            TestPoint { value: 1.0 },
            TestPoint { value: 2.0 },
            TestPoint { value: 3.0 },
            TestPoint { value: 4.0 },
            TestPoint { value: 5.0 },
        ];

        let vp_tree = VpTree::new(points);

        let target = TestPoint { value: 3.4 };
        let nearest = vp_tree.search_nearest_neighbor(&target).unwrap();

        assert_eq!(nearest.value, 3.0);
    }

    #[test]
    fn test_1() {
        #[derive(Debug)]
        struct TestPoint {
            value: f64,
        }

        impl Distance<TestPoint> for TestPoint {
            fn distance(&self, other: &TestPoint) -> f64 {
                (self.value - other.value).abs()
            }
        }

        let points = vec![
            TestPoint { value: 1.0 },
            TestPoint { value: 2.0 },
            TestPoint { value: 3.0 },
            TestPoint { value: 4.0 },
            TestPoint { value: 5.0 },
        ];

        let vp_tree = VpTree::new(points);

        let target = TestPoint { value: 3.4 };
        let nearest = vp_tree.search_closest_k_sorted(&target, 2).collect::<Vec<_>>();

        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].value, 3.0);
        assert_eq!(nearest[1].value, 4.0);
    }

    #[test]
    fn test_empty_tree() {
        #[derive(Debug)]
        struct TestPoint {
            value: f64,
        }
        impl Distance<TestPoint> for TestPoint {
            fn distance(&self, other: &TestPoint) -> f64 {
                (self.value - other.value).abs()
            }
        }
        let points: Vec<TestPoint> = vec![];
        let vp_tree = VpTree::new(points);

        let target = TestPoint { value: 3.5 };
        let nearest = vp_tree.search_closest_k_sorted(&target, 2).collect::<Vec<_>>();
        assert_eq!(nearest.len(), 0);
    }

    #[test]
    fn test_random_points() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestPoint {
            value: f64,
        }
        impl Distance<TestPoint> for TestPoint {
            fn distance(&self, other: &TestPoint) -> f64 {
                (self.value - other.value).abs()
            }
        }

        for _ in 0..10000 {
            let points: Vec<TestPoint> = (0..1000)
                .map(|_| TestPoint { value: fastrand::f64() * 1000.0 })
                .collect();
            
            let vp_tree = VpTree::new(points.clone());
            
            let target = TestPoint { value: 500.0 };
            let nearest = vp_tree.search_closest_k_sorted(&target, 10).collect::<Vec<_>>();
            
            let baseline_nearest = baseline_linear_search(&points, &target, 10);
            
            assert_eq!(nearest, baseline_nearest);
        }
    
    }

    #[test]
    fn search_in_radius_test() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestPoint {
            value: f64,
        }
        impl Distance<TestPoint> for TestPoint {
            fn distance(&self, other: &TestPoint) -> f64 {
                (self.value - other.value).abs()
            }
        }

        let points: Vec<TestPoint> = (0..1000)
            .map(|i| TestPoint { value: i as f64 })
            .collect();

        let vp_tree = VpTree::new(points.clone());

        let target = TestPoint { value: 500.0 };
        let radius = 10.0;
        let results: Vec<_> = vp_tree.search_in_radius_sorted(&target, radius).collect();

        let expected: Vec<_> = points
            .iter()
            .filter(|p| (p.value - target.value).abs() <= radius)
            .collect();

        assert_eq!(results.len(), expected.len());
        for r in results.iter() {
            assert!(expected.contains(r));
        }
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
}